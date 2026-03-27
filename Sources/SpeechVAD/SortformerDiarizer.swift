#if canImport(CoreML)
import CoreML
import Foundation
import AudioCommon

/// End-to-end neural speaker diarization using NVIDIA Sortformer (CoreML).
///
/// Sortformer directly predicts per-frame speaker activity for up to 4 speakers
/// without requiring separate embedding extraction or clustering. Runs on
/// Neural Engine at ~120x real-time.
///
/// ```swift
/// let diarizer = try await SortformerDiarizer.fromPretrained()
/// let result = diarizer.diarize(audio: samples, sampleRate: 16000)
/// for seg in result.segments {
///     print("Speaker \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
/// }
/// ```
public final class SortformerDiarizer {

    /// Default HuggingFace model ID for the CoreML Sortformer model
    public static let defaultModelId = "aufklarer/Sortformer-Diarization-CoreML"

    private let model: SortformerCoreMLModel
    private let melExtractor: SortformerMelExtractor
    let config: SortformerConfig

    /// Frame duration from model metadata (0.08s = 80ms per diarization frame)
    private let frameDuration: Float = 0.08

    // MARK: - Streaming State

    /// Speaker cache embeddings, flat `[spkcacheLen * fcDModel]`
    private var spkcache: [Float]
    /// Number of valid frames in speaker cache
    private var spkcacheLength: Int = 0
    /// Speaker cache predictions, flat `[spkcacheLen * maxSpeakers]` (for AOSC scoring)
    private var spkcachePreds: [Float]
    /// Whether spkcache predictions have been initialized (lazy init on first compression)
    private var hasSpkcachePreds: Bool = false
    /// FIFO buffer embeddings, flat `[fifoLen * fcDModel]`
    private var fifo: [Float]
    /// Number of valid frames in FIFO
    private var fifoLength: Int = 0
    /// FIFO buffer predictions, flat `[fifoLen * maxSpeakers]`
    private var fifoPreds: [Float]
    /// Running mean silence embedding, flat `[fcDModel]`
    private var meanSilEmb: [Float]
    /// Number of silence frames observed so far
    private var nSilFrames: Int = 0

    init(model: SortformerCoreMLModel, config: SortformerConfig = .default) {
        self.model = model
        self.config = config
        self.melExtractor = SortformerMelExtractor(config: config)
        self.spkcache = [Float](repeating: 0, count: config.spkcacheLen * config.fcDModel)
        self.spkcachePreds = [Float](repeating: 0, count: config.spkcacheLen * config.maxSpeakers)
        self.fifo = [Float](repeating: 0, count: config.fifoLen * config.fcDModel)
        self.fifoPreds = [Float](repeating: 0, count: config.fifoLen * config.maxSpeakers)
        self.meanSilEmb = [Float](repeating: 0, count: config.fcDModel)
    }

    /// Reset streaming state between different audio files.
    public func resetState() {
        spkcache = [Float](repeating: 0, count: config.spkcacheLen * config.fcDModel)
        spkcacheLength = 0
        spkcachePreds = [Float](repeating: 0, count: config.spkcacheLen * config.maxSpeakers)
        hasSpkcachePreds = false
        fifo = [Float](repeating: 0, count: config.fifoLen * config.fcDModel)
        fifoLength = 0
        fifoPreds = [Float](repeating: 0, count: config.fifoLen * config.maxSpeakers)
        meanSilEmb = [Float](repeating: 0, count: config.fcDModel)
        nSilFrames = 0
    }

    // MARK: - Loading

    /// Load a pre-trained Sortformer model from HuggingFace.
    ///
    /// - Parameters:
    ///   - modelId: HuggingFace model ID
    ///   - progressHandler: callback for download progress
    /// - Returns: ready-to-use diarizer
    public static func fromPretrained(
        modelId: String = defaultModelId,
        progressHandler: ((Double, String) -> Void)? = nil,
        useOfflineMode: Bool? = nil
    ) async throws -> SortformerDiarizer {
        progressHandler?(0.0, "Downloading Sortformer model...")

        let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)

        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: ["Sortformer.mlmodelc/**", "config.json"],
            progressHandler: { progress in
                progressHandler?(progress * 0.8, "Downloading Sortformer model...")
            },
            useOfflineMode: useOfflineMode
        )

        progressHandler?(0.8, "Loading CoreML model...")

        let modelURL = cacheDir.appendingPathComponent("Sortformer.mlmodelc", isDirectory: true)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId,
                reason: "CoreML model not found at \(modelURL.path)")
        }

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndNeuralEngine

        let mlModel: MLModel
        do {
            mlModel = try MLModel(contentsOf: modelURL, configuration: mlConfig)
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId,
                reason: "Failed to load CoreML model",
                underlying: error)
        }

        let config = SortformerConfig.default
        let coremlModel = SortformerCoreMLModel(model: mlModel, config: config)

        progressHandler?(1.0, "Ready")
        return SortformerDiarizer(model: coremlModel, config: config)
    }

    // MARK: - Diarization

    /// Run speaker diarization on complete audio.
    ///
    /// Processes audio in streaming chunks matching NeMo's streaming_feat_loader:
    /// each chunk is 112 mel frames = (leftCtx + coreChunk + rightCtx) × subsampling.
    /// Core predictions are extracted per chunk and concatenated.
    ///
    /// - Parameters:
    ///   - audio: PCM Float32 audio samples
    ///   - sampleRate: sample rate of the input audio
    ///   - config: optional override for diarization thresholds
    /// - Returns: diarization result with speaker-labeled segments
    public func diarize(
        audio: [Float],
        sampleRate: Int,
        config: DiarizationConfig? = nil,
        progressHandler: ((Double) -> Void)? = nil
    ) -> DiarizationResult {
        // Default to NeMo-optimized thresholds from SortformerConfig
        let config = config ?? DiarizationConfig(
            onset: self.config.onset,
            offset: self.config.offset,
            minSpeechDuration: self.config.minSpeechDuration,
            minSilenceDuration: self.config.minSilenceDuration)

        let samples = DiarizationHelpers.resample(audio, from: sampleRate, to: self.config.sampleRate)

        guard !samples.isEmpty else {
            return DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
        }

        resetState()

        // Extract mel features for the entire audio: [totalMelFrames, 128]
        let (melSpec, totalMelFrames) = melExtractor.extract(samples)

        guard totalMelFrames > 0 else {
            return DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
        }

        // Streaming chunking parameters (matching NeMo)
        let subFactor = self.config.subsamplingFactor
        let chunkLen = Int(self.config.chunkLenSeconds)  // 6 encoder output frames
        let leftCtx = Int(self.config.leftContextSeconds)  // 1
        let rightCtx = Int(self.config.rightContextSeconds) // 7
        let coreMelFrames = chunkLen * subFactor  // 48 mel frames per core chunk
        let coreMLInputFrames = 112  // Fixed CoreML input size
        let nMels = self.config.nMels
        let numSpeakers = self.config.maxSpeakers

        // Collect core predictions from each chunk
        var allChunkProbs = [[Float]]()  // Each entry: [coreFrames * numSpeakers]

        var sttFeat = 0
        var endFeat = 0

        while endFeat < totalMelFrames {
            if Task.isCancelled { break }
            let leftOffset = min(leftCtx * subFactor, sttFeat)
            endFeat = min(sttFeat + coreMelFrames, totalMelFrames)
            let rightOffset = min(rightCtx * subFactor, totalMelFrames - endFeat)

            let chunkStart = sttFeat - leftOffset
            let chunkEnd = endFeat + rightOffset
            let actualLen = chunkEnd - chunkStart

            // Build padded mel chunk [coreMLInputFrames, nMels]
            var chunkMel = [Float](repeating: 0, count: coreMLInputFrames * nMels)
            let framesToCopy = min(actualLen, coreMLInputFrames)
            for fi in 0..<framesToCopy {
                let srcBase = (chunkStart + fi) * nMels
                let dstBase = fi * nMels
                for di in 0..<nMels {
                    chunkMel[dstBase + di] = melSpec[srcBase + di]
                }
            }

            do {
                let output = try model.predict(
                    chunk: chunkMel,
                    chunkLength: actualLen,
                    spkcache: spkcache,
                    spkcacheLength: spkcacheLength,
                    fifo: fifo,
                    fifoLength: fifoLength
                )

                // Extract core predictions (skip spkcache + fifo + left context,
                // trim right context)
                let validEmbs: Int = output.validEmbFrames
                let lcFrames: Int = Int(Float(leftOffset) / Float(subFactor) + 0.5)
                let rcFrames: Int = Int(ceil(Float(rightOffset) / Float(subFactor)))
                let coreLen: Int = validEmbs - lcFrames - rcFrames
                let corePredLen = coreLen > 0 ? coreLen : 0

                let predOffset = spkcacheLength + fifoLength + lcFrames
                let totalPredFrames = output.predsFrames

                var chunkProbs = [Float]()
                for f in 0..<corePredLen {
                    let predFrame = predOffset + f
                    guard predFrame < totalPredFrames else { break }
                    for s in 0..<numSpeakers {
                        chunkProbs.append(output.pred(frame: predFrame, speaker: s))
                    }
                }
                allChunkProbs.append(chunkProbs)

                // Update streaming state (FIFO overflow → spkcache with AOSC)
                updateState(from: output, leftContext: lcFrames, rightContext: rcFrames)
            } catch {
                print("Warning: Sortformer inference failed on chunk at mel frame \(sttFeat): \(error)")
            }

            sttFeat = endFeat
            progressHandler?(Double(endFeat) / Double(totalMelFrames))
        }

        guard !allChunkProbs.isEmpty else {
            return DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
        }

        // Concatenate all core predictions
        let audioDuration = Float(samples.count) / Float(self.config.sampleRate)
        let segments = binarizeCorePredictions(
            allChunkProbs: allChunkProbs,
            audioDuration: audioDuration,
            numSpeakers: numSpeakers,
            onset: config.onset,
            offset: config.offset,
            minSpeechDuration: config.minSpeechDuration,
            minSilenceDuration: config.minSilenceDuration
        )

        let usedSpeakers = Set(segments.map(\.speakerId))
        return DiarizationResult(
            segments: segments,
            numSpeakers: usedSpeakers.count,
            speakerEmbeddings: []  // End-to-end model, no separate embeddings
        )
    }

    // MARK: - State Management (NeMo streaming_update with AOSC)

    /// Update spkcache and fifo buffers with AOSC compression.
    ///
    /// Follows NeMo's `streaming_update`:
    /// 1. Update FIFO predictions from model output (re-evaluated with new context)
    /// 2. Strip left/right context from chunk embeddings and predictions
    /// 3. Append core chunk to FIFO
    /// 4. If FIFO overflows, pop frames → update silence profile → append to spkcache
    /// 5. If spkcache exceeds capacity, compress via AOSC
    ///
    /// - Parameters:
    ///   - output: Model inference output containing embeddings and predictions
    ///   - leftContext: Number of left-context encoder frames in the chunk
    ///   - rightContext: Number of right-context encoder frames in the chunk
    private func updateState(from output: SortformerOutput, leftContext: Int, rightContext: Int) {
        let nSpk = config.maxSpeakers
        let dim = config.fcDModel
        let fifoCapacity = config.fifoLen
        let cacheCapacity = config.spkcacheLen

        let totalChunkFrames = output.validEmbFrames
        let coreFrames = totalChunkFrames - leftContext - rightContext
        guard coreFrames > 0 else { return }

        // ── 1. Update FIFO predictions from model output ──
        // The model re-evaluates FIFO frames with the new chunk's context,
        // giving improved predictions. Layout: [spkcache | fifo | chunk]
        let fifoPredStart = spkcacheLength * nSpk
        for f in 0..<fifoLength {
            let srcIdx = fifoPredStart + f * nSpk
            let dstIdx = f * nSpk
            for s in 0..<nSpk {
                fifoPreds[dstIdx + s] = output.speakerPreds[srcIdx + s]
            }
        }

        // ── 2. Extract core chunk embeddings and predictions (strip context) ──
        let coreEmbStart = leftContext * dim
        let corePredStart = (spkcacheLength + fifoLength + leftContext) * nSpk

        // ── 3. Append core chunk to FIFO ──
        let newFifoLength = fifoLength + coreFrames

        // Build combined FIFO (old + new) in temporary storage if overflow
        if newFifoLength <= fifoCapacity {
            // FIFO has room — just append
            for f in 0..<coreFrames {
                let srcEmbBase = coreEmbStart + f * dim
                let srcPredBase = corePredStart + f * nSpk
                let dstFrame = fifoLength + f
                let dstEmbBase = dstFrame * dim
                let dstPredBase = dstFrame * nSpk
                for d in 0..<dim {
                    fifo[dstEmbBase + d] = output.encoderEmbs[srcEmbBase + d]
                }
                for s in 0..<nSpk {
                    fifoPreds[dstPredBase + s] = output.speakerPreds[srcPredBase + s]
                }
            }
            fifoLength = newFifoLength
            return
        }

        // ── 4. FIFO overflow: pop → silence profile → spkcache ──

        // Build temp FIFO: [old fifo | new core chunk]
        var tempFifoEmb = [Float](repeating: 0, count: newFifoLength * dim)
        var tempFifoPred = [Float](repeating: 0, count: newFifoLength * nSpk)
        for i in 0..<(fifoLength * dim) { tempFifoEmb[i] = fifo[i] }
        for i in 0..<(fifoLength * nSpk) { tempFifoPred[i] = fifoPreds[i] }
        for f in 0..<coreFrames {
            let srcEmbBase = coreEmbStart + f * dim
            let srcPredBase = corePredStart + f * nSpk
            let dstFrame = fifoLength + f
            for d in 0..<dim {
                tempFifoEmb[dstFrame * dim + d] = output.encoderEmbs[srcEmbBase + d]
            }
            for s in 0..<nSpk {
                tempFifoPred[dstFrame * nSpk + s] = output.speakerPreds[srcPredBase + s]
            }
        }

        // Compute pop_out_len (matching NeMo)
        var popOutLen = config.spkcacheUpdatePeriod
        popOutLen = max(popOutLen, coreFrames - fifoCapacity + fifoLength)
        popOutLen = min(popOutLen, newFifoLength)

        // Extract pop-out frames
        let popOutEmbs = Array(tempFifoEmb[0..<(popOutLen * dim)])
        let popOutPreds = Array(tempFifoPred[0..<(popOutLen * nSpk)])

        // Update silence profile from popped frames
        AOSCCompressor.updateSilenceProfile(
            meanSilEmb: &meanSilEmb,
            nSilFrames: &nSilFrames,
            embSeq: popOutEmbs,
            preds: popOutPreds,
            nNewFrames: popOutLen,
            embDim: dim,
            nSpk: nSpk,
            silThreshold: config.silThreshold)

        // Trim FIFO: keep frames after pop-out
        let remainingFifo = newFifoLength - popOutLen
        for f in 0..<remainingFifo {
            let srcFrame = popOutLen + f
            for d in 0..<dim {
                fifo[f * dim + d] = tempFifoEmb[srcFrame * dim + d]
            }
            for s in 0..<nSpk {
                fifoPreds[f * nSpk + s] = tempFifoPred[srcFrame * nSpk + s]
            }
        }
        fifoLength = remainingFifo

        // ── 5. Append pop-out to spkcache, compress if needed ──
        let oldSpkcacheLength = spkcacheLength
        let newSpkcacheLength = spkcacheLength + popOutLen

        if newSpkcacheLength <= cacheCapacity {
            // Spkcache has room — just append
            for f in 0..<popOutLen {
                let dstFrame = spkcacheLength + f
                for d in 0..<dim {
                    spkcache[dstFrame * dim + d] = popOutEmbs[f * dim + d]
                }
            }
            if hasSpkcachePreds {
                for f in 0..<popOutLen {
                    let dstFrame = spkcacheLength + f
                    for s in 0..<nSpk {
                        spkcachePreds[dstFrame * nSpk + s] = popOutPreds[f * nSpk + s]
                    }
                }
            }
            spkcacheLength = newSpkcacheLength
        } else {
            // Spkcache overflows — compress with AOSC

            // Build combined input: [old spkcache | pop-out]
            var combinedEmbs = [Float](repeating: 0, count: newSpkcacheLength * dim)
            for i in 0..<(oldSpkcacheLength * dim) {
                combinedEmbs[i] = spkcache[i]
            }
            for f in 0..<popOutLen {
                let dstFrame = oldSpkcacheLength + f
                for d in 0..<dim {
                    combinedEmbs[dstFrame * dim + d] = popOutEmbs[f * dim + d]
                }
            }

            // Build combined predictions
            var combinedPreds = [Float](repeating: 0, count: newSpkcacheLength * nSpk)
            if hasSpkcachePreds {
                for i in 0..<(oldSpkcacheLength * nSpk) {
                    combinedPreds[i] = spkcachePreds[i]
                }
            } else {
                // First compression: get spkcache preds from model's current output
                for f in 0..<oldSpkcacheLength {
                    for s in 0..<nSpk {
                        combinedPreds[f * nSpk + s] = output.speakerPreds[f * nSpk + s]
                    }
                }
            }
            for f in 0..<popOutLen {
                let dstFrame = oldSpkcacheLength + f
                for s in 0..<nSpk {
                    combinedPreds[dstFrame * nSpk + s] = popOutPreds[f * nSpk + s]
                }
            }

            // Compress
            let result = AOSCCompressor.compress(
                embSeq: combinedEmbs,
                preds: combinedPreds,
                nFrames: newSpkcacheLength,
                meanSilEmb: meanSilEmb,
                config: config)

            // Update spkcache with compressed result
            for i in 0..<(cacheCapacity * dim) {
                spkcache[i] = result.spkcache[i]
            }
            for i in 0..<(cacheCapacity * nSpk) {
                spkcachePreds[i] = result.spkcachePreds[i]
            }
            spkcacheLength = cacheCapacity
            hasSpkcachePreds = true
        }
    }

    // MARK: - Binarization

    /// Concatenate per-chunk core predictions and binarize into segments.
    private func binarizeCorePredictions(
        allChunkProbs: [[Float]],
        audioDuration: Float,
        numSpeakers: Int,
        onset: Float,
        offset: Float,
        minSpeechDuration: Float,
        minSilenceDuration: Float
    ) -> [DiarizedSegment] {
        // Concatenate all chunk predictions into one flat array
        var allProbs = [Float]()
        for chunkProbs in allChunkProbs {
            allProbs.append(contentsOf: chunkProbs)
        }

        let totalFrames = allProbs.count / numSpeakers
        guard totalFrames > 0 else { return [] }

        // Apply sigmoid if predictions are logits
        for i in 0..<allProbs.count {
            if allProbs[i] > 1.0 || allProbs[i] < 0.0 {
                allProbs[i] = 1.0 / (1.0 + exp(-allProbs[i]))
            }
        }

        // Binarize each speaker track
        var allSegments = [DiarizedSegment]()

        for spk in 0..<numSpeakers {
            var probs = [Float](repeating: 0, count: totalFrames)
            for f in 0..<totalFrames {
                probs[f] = allProbs[f * numSpeakers + spk]
            }

            let rawSegments = PowersetDecoder.binarize(
                probs: probs,
                onset: onset,
                offset: offset,
                frameDuration: frameDuration
            )

            for seg in rawSegments {
                // Apply padOnset/padOffset (NeMo post-processing)
                let paddedStart = max(0, seg.startTime - config.padOnset)
                let paddedEnd = min(seg.endTime + config.padOffset, audioDuration)
                let duration = paddedEnd - paddedStart
                guard duration >= minSpeechDuration else { continue }
                allSegments.append(DiarizedSegment(
                    startTime: paddedStart,
                    endTime: paddedEnd,
                    speakerId: spk
                ))
            }
        }

        allSegments.sort { $0.startTime < $1.startTime }
        let merged = DiarizationHelpers.mergeSegments(allSegments, minSilence: minSilenceDuration)
        return DiarizationHelpers.compactSpeakerIds(merged)
    }
}

// MARK: - SpeakerDiarizationModel

extension SortformerDiarizer: SpeakerDiarizationModel {
    public var inputSampleRate: Int { config.sampleRate }

    public func diarize(audio: [Float], sampleRate: Int) -> [DiarizedSegment] {
        // Use NeMo-optimized thresholds from SortformerConfig (not generic DiarizationConfig)
        let cfg = DiarizationConfig(
            onset: config.onset,
            offset: config.offset,
            minSpeechDuration: config.minSpeechDuration,
            minSilenceDuration: config.minSilenceDuration)
        return diarize(audio: audio, sampleRate: sampleRate, config: cfg).segments
    }
}
#endif
