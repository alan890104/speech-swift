#if canImport(CoreML)
import Foundation

/// Shared streaming state for Sortformer diarization (spkcache + FIFO + AOSC).
///
/// Used by both `SortformerDiarizer` (batch) and `StreamingSortformerDiarizer` (streaming)
/// to avoid duplicating the `updateState()` logic.
struct SortformerStreamingState {

    /// Speaker cache embeddings, flat `[spkcacheLen * fcDModel]`
    private(set) var spkcache: [Float]
    /// Number of valid frames in speaker cache
    private(set) var spkcacheLength: Int = 0
    /// Speaker cache predictions, flat `[spkcacheLen * maxSpeakers]`
    private(set) var spkcachePreds: [Float]
    /// Whether spkcache predictions have been initialized (lazy init on first compression)
    private(set) var hasSpkcachePreds: Bool = false
    /// FIFO buffer embeddings, flat `[fifoLen * fcDModel]`
    private(set) var fifo: [Float]
    /// Number of valid frames in FIFO
    private(set) var fifoLength: Int = 0
    /// FIFO buffer predictions, flat `[fifoLen * maxSpeakers]`
    private(set) var fifoPreds: [Float]
    /// Running mean silence embedding, flat `[fcDModel]`
    private(set) var meanSilEmb: [Float]
    /// Number of silence frames observed so far
    private(set) var nSilFrames: Int = 0

    init(config: SortformerConfig) {
        self.spkcache = [Float](repeating: 0, count: config.spkcacheLen * config.fcDModel)
        self.spkcachePreds = [Float](repeating: 0, count: config.spkcacheLen * config.maxSpeakers)
        self.fifo = [Float](repeating: 0, count: config.fifoLen * config.fcDModel)
        self.fifoPreds = [Float](repeating: 0, count: config.fifoLen * config.maxSpeakers)
        self.meanSilEmb = [Float](repeating: 0, count: config.fcDModel)
    }

    /// Reset all state for a new audio session.
    mutating func reset(config: SortformerConfig) {
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

    /// Update spkcache and FIFO buffers with AOSC compression.
    ///
    /// Follows NeMo's `streaming_update`:
    /// 1. Update FIFO predictions from model output (re-evaluated with new context)
    /// 2. Strip left/right context from chunk embeddings and predictions
    /// 3. Append core chunk to FIFO
    /// 4. If FIFO overflows, pop frames → update silence profile → append to spkcache
    /// 5. If spkcache exceeds capacity, compress via AOSC
    mutating func update(from output: SortformerOutput, leftContext: Int, rightContext: Int, config: SortformerConfig) {
        let nSpk = config.maxSpeakers
        let dim = config.fcDModel
        let fifoCapacity = config.fifoLen
        let cacheCapacity = config.spkcacheLen

        let totalChunkFrames = output.validEmbFrames
        let coreFrames = totalChunkFrames - leftContext - rightContext
        guard coreFrames > 0 else { return }

        // ── 1. Update FIFO predictions from model output ──
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
}
#endif
