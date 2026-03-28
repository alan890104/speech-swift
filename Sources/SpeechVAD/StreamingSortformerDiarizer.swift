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
///     diarizer.process(samples: chunk)
/// }
/// let segments = diarizer.flush()
/// ```
public final class StreamingSortformerDiarizer {

    public static let defaultModelId = SortformerDiarizer.defaultModelId

    private let config: SortformerConfig
    private let frameDuration: Float = 0.08  // 80ms per diarization frame

    /// Shared chunk processing engine (mel extraction, inference, AOSC state)
    private var engine: SortformerChunkEngine
    /// Binarizer for converting probabilities to speaker segments
    private var binarizer: StreamingBinarizer
    /// Total diarization frames emitted so far (for timestamp calculation)
    private var totalDiarFrames: Int = 0
    /// Whether flush() has been called (prevents further process() calls)
    private var isFlushing: Bool = false

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
        self.config = config

        let sub = config.subsamplingFactor
        let rightCtxMel = Int(config.rightContextSeconds) * sub

        var minRightCtx = rightCtxMel
        if let la = lookahead {
            let laFrames = Int(la * Float(config.sampleRate) / Float(config.hopLength))
            minRightCtx = min(max(laFrames, 0), rightCtxMel)
        }

        self.engine = SortformerChunkEngine(
            model: model,
            melExtractor: SortformerMelExtractor(config: config),
            config: config,
            minRightContext: minRightCtx)

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

        feedBinarizer(engine.feedSamples(samples))
    }

    /// End of audio stream. Processes remaining audio, applies NeMo filtering,
    /// and returns all finalized segments.
    ///
    /// - Returns: Filtered, merged segments sorted by start time
    public func flush() -> [DiarizedSegment] {
        isFlushing = true

        feedBinarizer(engine.flush())

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
        totalDiarFrames = 0
        engine.reset()
        binarizer.reset()
    }

    /// Which speakers are currently active (speaking right now).
    public var activeSpeakers: [Int] {
        binarizer.activeSpeakers
    }

    // MARK: - Internal

    private func feedBinarizer(_ chunks: [ChunkPredictions]) {
        for chunk in chunks {
            // Clamp out-of-range values to sigmoid (defensive — model should output 0-1)
            var probs = chunk.probabilities
            for i in 0..<probs.count {
                if probs[i] > 1.0 || probs[i] < 0.0 {
                    probs[i] = 1.0 / (1.0 + exp(-probs[i]))
                }
            }
            let baseTime = Float(totalDiarFrames) * frameDuration
            binarizer.process(probs: probs, nFrames: chunk.coreFrameCount, baseTime: baseTime)
            totalDiarFrames += chunk.coreFrameCount
        }
    }
}
#endif
