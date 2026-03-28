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

    private var state: SortformerStreamingState

    init(model: SortformerCoreMLModel, config: SortformerConfig = .default) {
        self.model = model
        self.config = config
        self.melExtractor = SortformerMelExtractor(config: config)
        self.state = SortformerStreamingState(config: config)
    }

    /// Reset streaming state between different audio files.
    public func resetState() {
        state.reset(config: config)
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
        config: SortformerConfig = .default,
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
                    spkcache: state.spkcache,
                    spkcacheLength: state.spkcacheLength,
                    fifo: state.fifo,
                    fifoLength: state.fifoLength
                )

                // Extract core predictions (skip spkcache + fifo + left context,
                // trim right context)
                let validEmbs: Int = output.validEmbFrames
                let lcFrames: Int = Int(Float(leftOffset) / Float(subFactor) + 0.5)
                let rcFrames: Int = Int(ceil(Float(rightOffset) / Float(subFactor)))
                let coreLen: Int = validEmbs - lcFrames - rcFrames
                let corePredLen = coreLen > 0 ? coreLen : 0

                let predOffset = state.spkcacheLength + state.fifoLength + lcFrames
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
                state.update(from: output, leftContext: lcFrames, rightContext: rcFrames, config: config)
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

    // MARK: - Binarization

    /// Concatenate per-chunk core predictions and binarize into segments.
    ///
    /// Uses `StreamingBinarizer` which matches NeMo's exact two-step pipeline:
    /// `binarization()` (onset/offset + padding) → `filtering()` (short speech removal + gap filling).
    /// Processing is per-speaker, matching NeMo's behavior.
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

        // Use StreamingBinarizer — matches NeMo's binarization + filtering pipeline
        var binarizer = StreamingBinarizer(
            numSpeakers: numSpeakers,
            onset: onset,
            offset: offset,
            padOnset: config.padOnset,
            padOffset: config.padOffset,
            minDurationOn: minSpeechDuration,
            minDurationOff: minSilenceDuration,
            frameDuration: frameDuration)

        binarizer.process(probs: allProbs, nFrames: totalFrames, baseTime: 0)
        // endTime = last frame time, matching NeMo's `i * frame_length_in_sec` (i = N-1)
        let lastFrameTime = Float(totalFrames - 1) * frameDuration
        let segments = binarizer.flush(endTime: lastFrameTime)
        return DiarizationHelpers.compactSpeakerIds(segments)
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
