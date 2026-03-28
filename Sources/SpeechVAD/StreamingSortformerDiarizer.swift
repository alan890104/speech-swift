#if canImport(CoreML)
import CoreML
import Foundation
import AudioCommon

/// Fixed-memory streaming speaker diarizer using NVIDIA Sortformer with AOSC compression.
///
/// Accepts audio in arbitrary-size chunks and returns finalized diarization segments.
/// Memory usage is constant (~550 KB) regardless of total audio length.
///
/// ```swift
/// let diarizer = try await StreamingSortformerDiarizer.fromPretrained()
///
/// // From microphone, file reader, or any audio source:
/// while let chunk = getNextAudioChunk() {
///     let segments = diarizer.process(samples: chunk)
///     for seg in segments {
///         print("Speaker \(seg.speakerId): \(seg.startTime)s - \(seg.endTime)s")
///     }
/// }
/// let final = diarizer.finalize()
/// ```
public final class StreamingSortformerDiarizer {

    public static let defaultModelId = SortformerDiarizer.defaultModelId

    private let model: SortformerCoreMLModel
    private let melExtractor: SortformerMelExtractor
    private let config: SortformerConfig
    private let frameDuration: Float = 0.08  // 80ms per diarization frame

    // MARK: - Streaming Chunking Parameters

    /// Core mel frames per model inference step
    private let coreMelFrames: Int     // 48
    /// Left context mel frames
    private let leftCtxMel: Int        // 8
    /// Maximum right context mel frames (model's full capacity)
    private let rightCtxMel: Int       // 56
    /// Fixed CoreML input size
    private let coreMLInputFrames: Int // 112
    /// Minimum right context mel frames required before processing
    private let minRightCtxMel: Int

    // MARK: - Audio Buffering

    /// Accumulates incoming audio samples (resampled to 16kHz)
    private var audioBuffer: [Float] = []

    // MARK: - Mel Ring Buffer

    /// Extracted mel frames waiting to be processed, flat `[nMelFrames * nMels]`
    private var melBuffer: [Float] = []
    /// Number of mel frames in melBuffer
    private var melFrameCount: Int = 0
    /// Position of next core chunk start in mel frames (global, since session start)
    private var nextChunkStart: Int = 0
    /// Total mel frames extracted so far
    private var totalMelExtracted: Int = 0

    // MARK: - Model State (AOSC)

    private var state: SortformerStreamingState

    // MARK: - Binarization

    private var binarizer: StreamingBinarizer
    /// Total diarization frames emitted so far (for timestamp calculation)
    private var totalDiarFrames: Int = 0

    // MARK: - Init

    /// Create a streaming diarizer.
    ///
    /// - Parameters:
    ///   - model: CoreML Sortformer model
    ///   - config: Sortformer configuration
    ///   - lookahead: Lookahead duration in seconds. Controls the trade-off between
    ///     latency and accuracy. The model waits this long for future audio context
    ///     before producing results.
    ///     - `0.56` (default): Full right context. Maximum accuracy, matches batch `diarize()`.
    ///     - `0.0`: No lookahead. Minimum latency, reduced accuracy.
    init(model: SortformerCoreMLModel, config: SortformerConfig = .default, lookahead: Float? = nil) {
        self.model = model
        self.config = config
        self.melExtractor = SortformerMelExtractor(config: config)

        let sub = config.subsamplingFactor
        self.coreMelFrames = Int(config.chunkLenSeconds) * sub    // 6 * 8 = 48
        self.leftCtxMel = Int(config.leftContextSeconds) * sub     // 1 * 8 = 8
        self.rightCtxMel = Int(config.rightContextSeconds) * sub   // 7 * 8 = 56
        self.coreMLInputFrames = 112

        // Convert lookahead seconds to mel frames, clamped to model's maximum
        if let la = lookahead {
            let laFrames = Int(la * Float(config.sampleRate) / Float(config.hopLength))
            self.minRightCtxMel = min(max(laFrames, 0), rightCtxMel)
        } else {
            self.minRightCtxMel = rightCtxMel  // default: full context
        }

        self.state = SortformerStreamingState(config: config)

        self.binarizer = StreamingBinarizer(
            numSpeakers: config.maxSpeakers,
            onset: config.onset,
            offset: config.offset,
            padOnset: config.padOnset,
            padOffset: config.padOffset,
            minDurationOn: config.minSpeechDuration,
            minDurationOff: config.minSilenceDuration,
            frameDuration: frameDuration)
    }

    // MARK: - Loading

    /// Load a pre-trained Sortformer model from HuggingFace.
    ///
    /// - Parameters:
    ///   - modelId: HuggingFace model ID
    ///   - lookahead: Lookahead in seconds. `nil` = full context (0.56s, max accuracy).
    ///     `0` = no lookahead (min latency). Values between trade off latency vs accuracy.
    ///   - progressHandler: callback for download progress
    public static func fromPretrained(
        modelId: String = defaultModelId,
        config: SortformerConfig = .default,
        lookahead: Float? = nil,
        progressHandler: ((Double, String) -> Void)? = nil,
        useOfflineMode: Bool? = nil
    ) async throws -> StreamingSortformerDiarizer {
        progressHandler?(0.0, "Downloading Sortformer model...")

        let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: ["Sortformer.mlmodelc/**", "config.json"],
            progressHandler: { progress in
                progressHandler?(progress * 0.8, "Downloading Sortformer model...")
            },
            useOfflineMode: useOfflineMode)

        progressHandler?(0.8, "Loading CoreML model...")

        let modelURL = cacheDir.appendingPathComponent("Sortformer.mlmodelc", isDirectory: true)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId,
                reason: "CoreML model not found at \(modelURL.path)")
        }

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndNeuralEngine
        let mlModel = try MLModel(contentsOf: modelURL, configuration: mlConfig)

        let coremlModel = SortformerCoreMLModel(model: mlModel, config: config)

        progressHandler?(1.0, "Ready")
        return StreamingSortformerDiarizer(model: coremlModel, config: config, lookahead: lookahead)
    }

    // MARK: - Public API

    /// Push audio samples for processing.
    ///
    /// Samples can be any length — internally buffered to model chunk size.
    /// Segments accumulate internally; call `flush()` to get filtered output.
    ///
    /// - Parameter samples: PCM Float32 audio at 16kHz
    public func process(samples: [Float]) {
        precondition(!isFlushing,
            "process() called after flush(). Call resetState() before processing new audio.")

        // Extract new mel frames incrementally
        let newMel = melExtractor.extractIncremental(newSamples: samples)
        let newMelFrames = newMel.count / config.nMels
        guard newMelFrames > 0 else { return }

        melBuffer.append(contentsOf: newMel)
        melFrameCount += newMelFrames
        totalMelExtracted += newMelFrames

        // Process as many complete chunks as possible
        processAvailableChunks()
    }

    /// End of audio stream. Processes remaining audio, applies NeMo filtering,
    /// and returns all finalized segments.
    ///
    /// - Returns: Filtered, merged segments sorted by start time
    public func flush() -> [DiarizedSegment] {
        isFlushing = true

        // Extract remaining mel frames with right-side padding
        let finalMel = melExtractor.extractFinal()
        let finalFrames = finalMel.count / config.nMels
        if finalFrames > 0 {
            melBuffer.append(contentsOf: finalMel)
            melFrameCount += finalFrames
            totalMelExtracted += finalFrames
        }

        // Process any remaining chunks
        processAvailableChunks()

        // Handle the very last partial chunk if any mel frames remain
        if melFrameCount > 0 {
            processPartialChunk()
        }

        // Flush binarizer — applies NeMo filtering (short speech removal + gap filling)
        // endTime = last frame time, matching NeMo's `i * frame_length_in_sec` (i = N-1)
        let endTime = totalDiarFrames > 0 ? Float(totalDiarFrames - 1) * frameDuration : 0
        return binarizer.flush(endTime: endTime)
    }

    /// Run diarization on complete audio with constant memory.
    ///
    /// Same interface as `SortformerDiarizer.diarize()` but uses streaming
    /// internally — memory stays at ~550 KB regardless of audio length.
    /// Output is identical in format: sorted segments with compacted speaker IDs.
    ///
    /// - Parameters:
    ///   - audio: Complete PCM Float32 audio
    ///   - sampleRate: Sample rate of the input audio
    ///   - chunkSamples: Audio samples per processing chunk (default: 1 second)
    /// - Returns: Diarization result with speaker-labeled segments
    public func diarize(
        audio: [Float],
        sampleRate: Int,
        chunkSamples: Int = 16000,
        progressHandler: ((Double) -> Void)? = nil
    ) -> DiarizationResult {
        resetState()

        let samples = (sampleRate == config.sampleRate)
            ? audio
            : DiarizationHelpers.resample(audio, from: sampleRate, to: config.sampleRate)

        var offset = 0
        while offset < samples.count {
            let end = min(offset + chunkSamples, samples.count)
            let chunk = Array(samples[offset..<end])
            process(samples: chunk)
            offset = end
            progressHandler?(Double(offset) / Double(samples.count))
        }
        // flush() returns NeMo-filtered segments (short speech removed, gaps filled)
        let allSegments = flush()
        let compacted = DiarizationHelpers.compactSpeakerIds(allSegments)

        let usedSpeakers = Set(compacted.map(\.speakerId))
        return DiarizationResult(
            segments: compacted,
            numSpeakers: usedSpeakers.count,
            speakerEmbeddings: [])
    }

    /// Reset all state for a new audio session.
    public func resetState() {
        isFlushing = false
        audioBuffer = []
        melBuffer = []
        melFrameCount = 0
        nextChunkStart = 0
        totalMelExtracted = 0
        totalDiarFrames = 0

        melExtractor.resetStreamingState()
        state.reset(config: config)

        binarizer.reset()
    }

    /// Which speakers are currently active (speaking right now).
    public var activeSpeakers: [Int] {
        binarizer.activeSpeakers
    }

    // MARK: - Internal: Chunk Processing

    /// Whether we're in flush mode (finalize called, accept partial right context).
    private var isFlushing: Bool = false

    /// Process all complete chunks available in the mel buffer.
    @discardableResult
    private func processAvailableChunks() -> [DiarizedSegment] {
        let nMels = config.nMels
        let numSpeakers = config.maxSpeakers
        let subFactor = config.subsamplingFactor

        while true {
            // Calculate context for this chunk
            let sttFeat = nextChunkStart
            let endFeat = sttFeat + coreMelFrames

            // We need enough mel frames for core + right context
            let rightAvailable = totalMelExtracted - endFeat
            let leftCtx = min(leftCtxMel, sttFeat)

            // During normal streaming: require minimum right context before processing.
            // During flush: accept whatever right context is available.
            guard endFeat <= totalMelExtracted else { break }
            if !isFlushing {
                guard rightAvailable >= minRightCtxMel else { break }
            }
            let rightCtx = min(rightCtxMel, max(0, rightAvailable))

            let neededInBuffer = (endFeat + rightCtx) - (totalMelExtracted - melFrameCount)
            guard neededInBuffer <= melFrameCount else { break }

            let chunkStart = sttFeat - leftCtx
            let chunkEnd = endFeat + rightCtx
            let actualLen = chunkEnd - chunkStart

            // Build padded mel chunk [coreMLInputFrames, nMels]
            var chunkMel = [Float](repeating: 0, count: coreMLInputFrames * nMels)

            // Map global mel frame indices to melBuffer local indices
            let bufferGlobalStart = totalMelExtracted - melFrameCount
            let localStart = chunkStart - bufferGlobalStart
            let framesToCopy = min(actualLen, coreMLInputFrames)

            for fi in 0..<framesToCopy {
                let srcFrame = localStart + fi
                guard srcFrame >= 0 && srcFrame < melFrameCount else { continue }
                let srcBase = srcFrame * nMels
                let dstBase = fi * nMels
                for di in 0..<nMels {
                    chunkMel[dstBase + di] = melBuffer[srcBase + di]
                }
            }

            // Run model inference
            do {
                let output = try model.predict(
                    chunk: chunkMel,
                    chunkLength: actualLen,
                    spkcache: state.spkcache,
                    spkcacheLength: state.spkcacheLength,
                    fifo: state.fifo,
                    fifoLength: state.fifoLength)

                // Extract core predictions
                let validEmbs = output.validEmbFrames
                let lcFrames = Int(Float(leftCtx) / Float(subFactor) + 0.5)
                let rcFrames = Int(ceil(Float(rightCtx) / Float(subFactor)))
                let coreLen = max(0, validEmbs - lcFrames - rcFrames)
                let predOffset = state.spkcacheLength + state.fifoLength + lcFrames

                // Feed core predictions to binarizer
                if coreLen > 0 {
                    var coreProbs = [Float](repeating: 0, count: coreLen * numSpeakers)
                    for f in 0..<coreLen {
                        let predFrame = predOffset + f
                        guard predFrame < output.predsFrames else { break }
                        for s in 0..<numSpeakers {
                            var prob = output.pred(frame: predFrame, speaker: s)
                            if prob > 1.0 || prob < 0.0 {
                                prob = 1.0 / (1.0 + exp(-prob))
                            }
                            coreProbs[f * numSpeakers + s] = prob
                        }
                    }

                    let baseTime = Float(totalDiarFrames) * frameDuration
                    binarizer.process(
                        probs: coreProbs, nFrames: coreLen, baseTime: baseTime)
                    totalDiarFrames += coreLen
                }

                // Update streaming state (FIFO + AOSC)
                state.update(from: output, leftContext: lcFrames, rightContext: rcFrames, config: config)

            } catch {
                // Skip failed chunk
            }

            nextChunkStart = endFeat

            // Trim consumed mel frames from buffer
            let bufferTrimTo = nextChunkStart - (totalMelExtracted - melFrameCount)
            if bufferTrimTo > 0 && bufferTrimTo <= melFrameCount {
                melBuffer.removeFirst(bufferTrimTo * nMels)
                melFrameCount -= bufferTrimTo
            }
        }

        return []  // segments accumulated in binarizer, returned by flush()
    }

    /// Process a partial final chunk (fewer than coreMelFrames).
    @discardableResult
    private func processPartialChunk() -> [DiarizedSegment] {
        let nMels = config.nMels
        let numSpeakers = config.maxSpeakers
        let subFactor = config.subsamplingFactor

        let sttFeat = nextChunkStart
        let remainingMel = totalMelExtracted - sttFeat
        guard remainingMel > 0 else { return [] }

        let leftCtx = min(leftCtxMel, sttFeat)
        let chunkStart = sttFeat - leftCtx
        let chunkEnd = totalMelExtracted
        let actualLen = chunkEnd - chunkStart

        var chunkMel = [Float](repeating: 0, count: coreMLInputFrames * nMels)
        let bufferGlobalStart = totalMelExtracted - melFrameCount
        let localStart = chunkStart - bufferGlobalStart
        let framesToCopy = min(actualLen, coreMLInputFrames)

        for fi in 0..<framesToCopy {
            let srcFrame = localStart + fi
            guard srcFrame >= 0 && srcFrame < melFrameCount else { continue }
            let srcBase = srcFrame * nMels
            let dstBase = fi * nMels
            for di in 0..<nMels {
                chunkMel[dstBase + di] = melBuffer[srcBase + di]
            }
        }

        do {
            let output = try model.predict(
                chunk: chunkMel,
                chunkLength: actualLen,
                spkcache: state.spkcache,
                spkcacheLength: state.spkcacheLength,
                fifo: state.fifo,
                fifoLength: state.fifoLength)

            let validEmbs = output.validEmbFrames
            let lcFrames = Int(Float(leftCtx) / Float(subFactor) + 0.5)
            let coreLen = max(0, validEmbs - lcFrames)
            let predOffset = state.spkcacheLength + state.fifoLength + lcFrames

            if coreLen > 0 {
                var coreProbs = [Float](repeating: 0, count: coreLen * numSpeakers)
                for f in 0..<coreLen {
                    let predFrame = predOffset + f
                    guard predFrame < output.predsFrames else { break }
                    for s in 0..<numSpeakers {
                        var prob = output.pred(frame: predFrame, speaker: s)
                        if prob > 1.0 || prob < 0.0 {
                            prob = 1.0 / (1.0 + exp(-prob))
                        }
                        coreProbs[f * numSpeakers + s] = prob
                    }
                }
                let baseTime = Float(totalDiarFrames) * frameDuration
                binarizer.process(probs: coreProbs, nFrames: coreLen, baseTime: baseTime)
                totalDiarFrames += coreLen
            }

            state.update(from: output, leftContext: lcFrames, rightContext: 0, config: config)
            melFrameCount = 0
            melBuffer = []
            return []
        } catch {
            return []
        }
    }
}
#endif
