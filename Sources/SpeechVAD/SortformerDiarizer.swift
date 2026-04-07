#if canImport(CoreML)
import CoreML
import Foundation
import AudioCommon

/// Single diarization frame with per-speaker activity probabilities (80ms resolution).
public struct SpeakerFrame: Sendable {
    /// Absolute time in seconds, measured from the most recent `resetState()` call
    public let time: Float
    /// Per-speaker activity probabilities [maxSpeakers=4], each 0.0~1.0
    public let probabilities: [Float]
}

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
    public static let defaultModelId = "Alkd/Sortformer-Diarization-CoreML"

    private let model: SortformerCoreMLModel
    let config: SortformerConfig

    /// Frame duration from model metadata (0.08s = 80ms per diarization frame)
    private let frameDuration: Float = 0.08

    // MARK: - State

    /// Batch-mode model state (spkcache, FIFO, AOSC) — used by diarize()
    private var state: SortformerStreamingState
    /// Streaming chunk engine — used by process()/flush()
    private var engine: SortformerChunkEngine
    /// Whether flush() has been called (prevents further process() calls)
    private var streamFlushed: Bool = false

    init(model: SortformerCoreMLModel, config: SortformerConfig = .default) {
        self.model = model
        self.config = config
        self.state = SortformerStreamingState(config: config)
        self.engine = SortformerChunkEngine(
            model: model,
            melExtractor: SortformerMelExtractor(config: config),
            config: config)
    }

    /// Reset all streaming state for a new audio session.
    ///
    /// Clears model state (spkcache, FIFO, AOSC) and incremental mel extraction state.
    /// Call before starting a new `process()`/`flush()` cycle.
    public func resetState() {
        state.reset(config: config)
        engine.reset()
        streamFlushed = false
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

    // MARK: - Incremental Processing

    /// Incrementally process new audio samples.
    ///
    /// Extracts mel features, runs CoreML inference when enough frames accumulate,
    /// and returns per-frame speaker activity probabilities.
    ///
    /// Call repeatedly with streaming audio chunks (e.g., 1 second at a time).
    /// State (spkcache, fifo) is preserved across calls for consistent speaker IDs.
    ///
    /// Returns empty array during warmup (first ~1s, before enough mel frames for one chunk).
    ///
    /// - Parameters:
    ///   - samples: PCM Float32 audio samples
    ///   - sampleRate: sample rate of the input audio (default: 16000)
    /// - Returns: per-frame speaker activity probabilities for newly processed frames
    public func process(samples: [Float], sampleRate: Int = 16000) -> [SpeakerFrame] {
        precondition(!streamFlushed,
            "process() called after flush(). Call resetState() before processing new audio.")
        guard !samples.isEmpty else { return [] }
        let resampled = (sampleRate == config.sampleRate)
            ? samples
            : DiarizationHelpers.resample(samples, from: sampleRate, to: config.sampleRate)
        return chunksToFrames(engine.feedSamples(resampled))
    }

    /// Finalize: process any remaining buffered mel frames.
    ///
    /// Call at end of stream. Returns the final batch of `SpeakerFrame`s.
    /// After calling this, call `resetState()` before processing new audio.
    public func flush() -> [SpeakerFrame] {
        streamFlushed = true
        return chunksToFrames(engine.flush())
    }

    private func chunksToFrames(_ chunks: [ChunkPredictions]) -> [SpeakerFrame] {
        let numSpeakers = config.maxSpeakers
        var frames = [SpeakerFrame]()
        for chunk in chunks {
            for f in 0..<chunk.coreFrameCount {
                let frameIndex = chunk.startFrameIndex + f
                let time = Float(frameIndex) * frameDuration
                var probs = [Float](repeating: 0, count: numSpeakers)
                for s in 0..<numSpeakers {
                    probs[s] = chunk.probabilities[f * numSpeakers + s]
                }
                frames.append(SpeakerFrame(time: time, probabilities: probs))
            }
        }
        return frames
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
        let diarConfig = config ?? DiarizationConfig(
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
        let (melSpec, totalMelFrames) = engine.melExtractor.extract(samples)

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
                let output = try autoreleasepool {
                    try model.predict(
                        chunk: chunkMel,
                        chunkLength: actualLen,
                        spkcache: state.spkcache,
                        spkcacheLength: state.spkcacheLength,
                        fifo: state.fifo,
                        fifoLength: state.fifoLength
                    )
                }

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
                state.update(from: output, leftContext: lcFrames, rightContext: rcFrames, config: self.config)
            } catch {
                AudioLog.inference.warning("Sortformer inference failed at mel frame \(sttFeat): \(error.localizedDescription)")
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
            onset: diarConfig.onset,
            offset: diarConfig.offset,
            minSpeechDuration: diarConfig.minSpeechDuration,
            minSilenceDuration: diarConfig.minSilenceDuration
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
