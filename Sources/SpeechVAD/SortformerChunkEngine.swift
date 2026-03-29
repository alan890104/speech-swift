#if canImport(CoreML)
import Foundation
import AudioCommon

/// Core predictions from one processed chunk.
struct ChunkPredictions {
    /// Number of core prediction frames
    let coreFrameCount: Int
    /// Flat probability array [coreFrameCount * numSpeakers], values 0.0~1.0
    let probabilities: [Float]
    /// Global index of the first core frame (for time/offset calculation)
    let startFrameIndex: Int
}

/// Internal streaming chunk processing engine for Sortformer.
///
/// Handles mel buffering, chunk boundary calculation, model inference,
/// and AOSC state management. Callers wrap this to produce their desired
/// output format (`SpeakerFrame`, `DiarizedSegment`, etc.).
struct SortformerChunkEngine {
    private let model: SortformerCoreMLModel
    let melExtractor: SortformerMelExtractor
    let config: SortformerConfig

    // Chunking parameters
    private let coreMelFrames: Int     // 48
    private let leftCtxMel: Int        // 8
    private let rightCtxMel: Int       // 56
    private let minRightCtxMel: Int
    private let coreMLInputFrames: Int = 112

    // Mel buffer
    private var melBuffer: [Float] = []
    private var melFrameCount: Int = 0
    private var nextChunkStart: Int = 0
    private var totalMelExtracted: Int = 0

    // Model state (spkcache, FIFO, AOSC)
    private var state: SortformerStreamingState

    // Output tracking
    private(set) var totalCoreFramesEmitted: Int = 0

    /// - Parameters:
    ///   - model: CoreML Sortformer model
    ///   - melExtractor: Mel feature extractor (shared with caller for batch use)
    ///   - config: Sortformer configuration
    ///   - minRightContext: Minimum right context mel frames before processing.
    ///     `nil` = full right context (56 frames). `0` = no lookahead.
    init(model: SortformerCoreMLModel, melExtractor: SortformerMelExtractor,
         config: SortformerConfig, minRightContext: Int? = nil) {
        self.model = model
        self.melExtractor = melExtractor
        self.config = config

        let sub = config.subsamplingFactor
        self.coreMelFrames = Int(config.chunkLenSeconds) * sub
        self.leftCtxMel = Int(config.leftContextSeconds) * sub
        self.rightCtxMel = Int(config.rightContextSeconds) * sub
        self.minRightCtxMel = minRightContext ?? self.rightCtxMel

        self.state = SortformerStreamingState(config: config)
    }

    mutating func reset() {
        melBuffer = []
        melFrameCount = 0
        nextChunkStart = 0
        totalMelExtracted = 0
        totalCoreFramesEmitted = 0
        melExtractor.resetStreamingState()
        state.reset(config: config)
    }

    /// Feed new pre-resampled audio samples (16kHz).
    /// Returns core predictions for any completed chunks.
    mutating func feedSamples(_ samples: [Float]) -> [ChunkPredictions] {
        guard !samples.isEmpty else { return [] }

        let newMel = melExtractor.extractIncremental(newSamples: samples)
        let newMelFrames = newMel.count / config.nMels
        guard newMelFrames > 0 else { return [] }

        melBuffer.append(contentsOf: newMel)
        melFrameCount += newMelFrames
        totalMelExtracted += newMelFrames

        return processAvailableChunks(isFlushing: false)
    }

    /// Finalize stream: extract remaining mel frames and process all remaining chunks.
    mutating func flush() -> [ChunkPredictions] {
        let finalMel = melExtractor.extractFinal()
        let finalFrames = finalMel.count / config.nMels
        if finalFrames > 0 {
            melBuffer.append(contentsOf: finalMel)
            melFrameCount += finalFrames
            totalMelExtracted += finalFrames
        }

        var results = processAvailableChunks(isFlushing: true)

        if melFrameCount > 0 {
            if let partial = processPartialChunk() {
                results.append(partial)
            }
        }

        return results
    }

    // MARK: - Internal

    private mutating func processAvailableChunks(isFlushing: Bool) -> [ChunkPredictions] {
        let nMels = config.nMels
        let numSpeakers = config.maxSpeakers
        let subFactor = config.subsamplingFactor

        var results = [ChunkPredictions]()

        while true {
            if Task.isCancelled { break }

            let sttFeat = nextChunkStart
            let endFeat = sttFeat + coreMelFrames

            guard endFeat <= totalMelExtracted else { break }

            let rightAvailable = totalMelExtracted - endFeat
            if !isFlushing {
                guard rightAvailable >= minRightCtxMel else { break }
            }

            let leftCtx = min(leftCtxMel, sttFeat)
            let rightCtx = min(rightCtxMel, rightAvailable)

            let chunkStart = sttFeat - leftCtx
            let chunkEnd = endFeat + rightCtx
            let actualLen = chunkEnd - chunkStart

            // Build padded mel chunk [coreMLInputFrames, nMels]
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
                // autoreleasepool drains CoreML's MLMultiArray objects each iteration,
                // preventing unbounded memory growth during long audio processing.
                let output = try autoreleasepool {
                    try model.predict(
                        chunk: chunkMel,
                        chunkLength: actualLen,
                        spkcache: state.spkcache,
                        spkcacheLength: state.spkcacheLength,
                        fifo: state.fifo,
                        fifoLength: state.fifoLength)
                }

                let validEmbs = output.validEmbFrames
                let lcFrames = Int(Float(leftCtx) / Float(subFactor) + 0.5)
                let rcFrames = Int(ceil(Float(rightCtx) / Float(subFactor)))
                let coreLen = max(0, validEmbs - lcFrames - rcFrames)
                let predOffset = state.spkcacheLength + state.fifoLength + lcFrames

                if coreLen > 0 {
                    var probs = [Float](repeating: 0, count: coreLen * numSpeakers)
                    for f in 0..<coreLen {
                        let predFrame = predOffset + f
                        guard predFrame < output.predsFrames else { break }
                        for s in 0..<numSpeakers {
                            probs[f * numSpeakers + s] = output.pred(frame: predFrame, speaker: s)
                        }
                    }
                    results.append(ChunkPredictions(
                        coreFrameCount: coreLen,
                        probabilities: probs,
                        startFrameIndex: totalCoreFramesEmitted))
                    totalCoreFramesEmitted += coreLen
                }

                state.update(from: output, leftContext: lcFrames, rightContext: rcFrames, config: config)
            } catch {
                AudioLog.inference.warning("Sortformer inference failed at mel frame \(sttFeat): \(error.localizedDescription)")
            }

            nextChunkStart = endFeat

            // Trim consumed mel frames, keeping leftCtxMel for next chunk's left context
            let bufGlobalStart = totalMelExtracted - melFrameCount
            let trimCount = max(0, nextChunkStart - leftCtxMel - bufGlobalStart)
            if trimCount > 0 && trimCount < melFrameCount {
                melBuffer.removeFirst(trimCount * nMels)
                melFrameCount -= trimCount
            }
        }

        return results
    }

    private mutating func processPartialChunk() -> ChunkPredictions? {
        let nMels = config.nMels
        let numSpeakers = config.maxSpeakers
        let subFactor = config.subsamplingFactor

        let sttFeat = nextChunkStart
        let remainingMel = totalMelExtracted - sttFeat
        guard remainingMel > 0 else { return nil }

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
            let output = try autoreleasepool {
                try model.predict(
                    chunk: chunkMel,
                    chunkLength: actualLen,
                    spkcache: state.spkcache,
                    spkcacheLength: state.spkcacheLength,
                    fifo: state.fifo,
                    fifoLength: state.fifoLength)
            }

            let validEmbs = output.validEmbFrames
            let lcFrames = Int(Float(leftCtx) / Float(subFactor) + 0.5)
            let coreLen = max(0, validEmbs - lcFrames)
            let predOffset = state.spkcacheLength + state.fifoLength + lcFrames

            guard coreLen > 0 else { return nil }

            var probs = [Float](repeating: 0, count: coreLen * numSpeakers)
            for f in 0..<coreLen {
                let predFrame = predOffset + f
                guard predFrame < output.predsFrames else { break }
                for s in 0..<numSpeakers {
                    probs[f * numSpeakers + s] = output.pred(frame: predFrame, speaker: s)
                }
            }

            let result = ChunkPredictions(
                coreFrameCount: coreLen,
                probabilities: probs,
                startFrameIndex: totalCoreFramesEmitted)
            totalCoreFramesEmitted += coreLen

            state.update(from: output, leftContext: lcFrames, rightContext: 0, config: config)

            melBuffer = []
            melFrameCount = 0

            return result
        } catch {
            AudioLog.inference.warning("Sortformer inference failed on partial chunk at mel frame \(sttFeat): \(error.localizedDescription)")
            return nil
        }
    }
}
#endif
