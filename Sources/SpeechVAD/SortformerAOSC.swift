import Foundation

/// AOSC (Arrival-Order Speaker Cache) compression for Sortformer streaming diarization.
///
/// Implements the frame selection algorithm from NeMo's `SortformerModules._compress_spkcache`:
/// selects the most important frames from the speaker cache based on prediction scores,
/// ensuring balanced speaker representation and temporal freshness.
///
/// Reference: "Streaming Sortformer: Speaker Cache-Based Online Speaker Diarization
/// with Arrival-Time Ordering" (arXiv:2507.18446)
enum AOSCCompressor {

    /// Result of AOSC compression.
    struct Result {
        /// Compressed speaker cache embeddings, flat `[spkcacheLen * embDim]`
        let spkcache: [Float]
        /// Corresponding predictions, flat `[spkcacheLen * nSpk]`
        let spkcachePreds: [Float]
    }

    // MARK: - Main Compression

    /// Compress speaker cache by selecting the most important frames.
    ///
    /// Matches NeMo's `_compress_spkcache` for inference (no speaker permutation, no random noise).
    ///
    /// - Parameters:
    ///   - embSeq: Embeddings to compress, flat `[nFrames * embDim]`
    ///   - preds: Speaker predictions, flat `[nFrames * nSpk]`
    ///   - nFrames: Number of input frames (must be > spkcacheLen)
    ///   - meanSilEmb: Mean silence embedding, flat `[embDim]`
    ///   - config: Sortformer configuration with AOSC parameters
    /// - Returns: Compressed cache of exactly `spkcacheLen` frames
    static func compress(
        embSeq: [Float],
        preds: [Float],
        nFrames: Int,
        meanSilEmb: [Float],
        config: SortformerConfig
    ) -> Result {
        let nSpk = config.maxSpeakers
        let spkcacheLen = config.spkcacheLen
        let silFrames = config.spkcacheSilFramesPerSpk
        let spkcacheLenPerSpk = spkcacheLen / nSpk - silFrames
        let strongBoostPerSpk = Int(floor(Float(spkcacheLenPerSpk) * config.strongBoostRate))
        let weakBoostPerSpk = Int(floor(Float(spkcacheLenPerSpk) * config.weakBoostRate))
        let minPosScoresPerSpk = Int(floor(Float(spkcacheLenPerSpk) * config.minPosScoresRate))

        // Step 1: Compute log prediction scores
        var scores = getLogPredScores(
            preds: preds, nFrames: nFrames, nSpk: nSpk,
            threshold: config.predScoreThreshold)

        // Step 2: Disable non-speech and overlapped-speech scores
        disableLowScores(
            preds: preds, scores: &scores, nFrames: nFrames, nSpk: nSpk,
            minPosScoresPerSpk: minPosScoresPerSpk)

        // Step 3: Boost latest frames (beyond original spkcache_len)
        if config.scoresBoostLatest > 0 {
            let boost = config.scoresBoostLatest
            for f in spkcacheLen..<nFrames {
                let base = f * nSpk
                for s in 0..<nSpk where scores[base + s].isFinite {
                    scores[base + s] += boost
                }
            }
        }

        // Step 4: Strong boosting (ensure each speaker gets minimum representation)
        boostTopkScores(
            scores: &scores, nFrames: nFrames, nSpk: nSpk,
            nBoostPerSpk: strongBoostPerSpk, scaleFactor: 2.0)

        // Step 5: Weak boosting (prevent single-speaker dominance)
        boostTopkScores(
            scores: &scores, nFrames: nFrames, nSpk: nSpk,
            nBoostPerSpk: weakBoostPerSpk, scaleFactor: 1.0)

        // Step 6: Append silence frame placeholders with inf scores
        let nFramesWithSil = nFrames + silFrames
        if silFrames > 0 {
            scores.reserveCapacity(nFramesWithSil * nSpk)
            for _ in 0..<(silFrames * nSpk) {
                scores.append(Float.infinity)
            }
        }

        // Step 7: Select top-K frame indices globally across speakers
        let (topkIndices, isDisabled) = getTopkIndices(
            scores: scores, nFrames: nFramesWithSil, nSpk: nSpk,
            spkcacheLen: spkcacheLen, silFramesPerSpk: silFrames)

        // Step 8: Gather embeddings and predictions at selected indices
        return gatherSpkcacheAndPreds(
            embSeq: embSeq, preds: preds,
            topkIndices: topkIndices, isDisabled: isDisabled,
            meanSilEmb: meanSilEmb,
            spkcacheLen: spkcacheLen, embDim: config.fcDModel, nSpk: nSpk)
    }

    // MARK: - Score Computation

    /// Compute per-frame importance scores based on speaker prediction probabilities.
    ///
    /// Score formula: `log(P_i) - log(1-P_i) + Σ_j[log(1-P_j)] - log(0.5)`
    ///
    /// High scores indicate confident, non-overlapped speech for speaker i.
    /// Matches NeMo's `_get_log_pred_scores`.
    static func getLogPredScores(
        preds: [Float], nFrames: Int, nSpk: Int, threshold: Float
    ) -> [Float] {
        var scores = [Float](repeating: 0, count: nFrames * nSpk)
        let logHalf = logf(0.5)

        for f in 0..<nFrames {
            let base = f * nSpk

            // Compute sum of log(1-P_j) across all speakers
            var log1ProbsSum: Float = 0
            for s in 0..<nSpk {
                log1ProbsSum += logf(max(1.0 - preds[base + s], threshold))
            }

            for s in 0..<nSpk {
                let p = preds[base + s]
                let logP = logf(max(p, threshold))
                let log1P = logf(max(1.0 - p, threshold))
                scores[base + s] = logP - log1P + log1ProbsSum - logHalf
            }
        }

        return scores
    }

    /// Disable scores for non-speech frames and overlapped-speech frames.
    ///
    /// - Non-speech: prediction ≤ 0.5 → score becomes -inf
    /// - Overlapped speech (non-positive score): disabled if the speaker has
    ///   at least `minPosScoresPerSpk` positive-scored frames
    ///
    /// Matches NeMo's `_disable_low_scores`.
    static func disableLowScores(
        preds: [Float], scores: inout [Float], nFrames: Int, nSpk: Int,
        minPosScoresPerSpk: Int
    ) {
        // Pass 1: Disable non-speech, count positive scores per speaker
        var posCountPerSpk = [Int](repeating: 0, count: nSpk)
        for f in 0..<nFrames {
            let base = f * nSpk
            for s in 0..<nSpk {
                if preds[base + s] <= 0.5 {
                    scores[base + s] = -.infinity
                } else if scores[base + s] > 0 {
                    posCountPerSpk[s] += 1
                }
            }
        }

        // Pass 2: Disable non-positive scores for speakers with enough positive frames
        for f in 0..<nFrames {
            let base = f * nSpk
            for s in 0..<nSpk {
                let isSpeech = preds[base + s] > 0.5
                let isPos = scores[base + s] > 0
                if !isPos && isSpeech && posCountPerSpk[s] >= minPosScoresPerSpk {
                    scores[base + s] = -.infinity
                }
            }
        }
    }

    /// Boost the top-K scoring frames per speaker to ensure balanced representation.
    ///
    /// For each speaker independently, finds the K highest-scored frames and
    /// adds `scaleFactor * log(2)` to their scores. Frames with -inf scores
    /// are never boosted.
    ///
    /// Matches NeMo's `_boost_topk_scores`.
    static func boostTopkScores(
        scores: inout [Float], nFrames: Int, nSpk: Int,
        nBoostPerSpk: Int, scaleFactor: Float, offset: Float = 0.5
    ) {
        guard nBoostPerSpk > 0 else { return }
        let boost = -scaleFactor * logf(offset)

        for s in 0..<nSpk {
            // Collect (score, frameIndex) for this speaker, sort descending by score
            // then ascending by frame index for deterministic tie-breaking
            var indexed = [(score: Float, frame: Int)]()
            indexed.reserveCapacity(nFrames)
            for f in 0..<nFrames {
                indexed.append((scores[f * nSpk + s], f))
            }
            indexed.sort { a, b in
                if a.score != b.score { return a.score > b.score }
                return a.frame < b.frame
            }

            // Boost top-K (skip -inf)
            let k = min(nBoostPerSpk, indexed.count)
            for i in 0..<k {
                let f = indexed[i].frame
                let idx = f * nSpk + s
                if scores[idx].isFinite {
                    scores[idx] += boost
                }
            }
        }
    }

    // MARK: - Frame Selection

    /// Select the top `spkcacheLen` frame indices globally across all speakers.
    ///
    /// Scores are flattened as `(speaker, frame)` — speaker 0's frames first,
    /// then speaker 1's, etc. After selection, indices are sorted to preserve
    /// chronological order within each speaker block (arrival-time ordering).
    ///
    /// Silence placeholder frames (appended at the end) are marked as disabled
    /// and replaced with mean silence embedding during gathering.
    ///
    /// Matches NeMo's `_get_topk_indices`.
    static func getTopkIndices(
        scores: [Float], nFrames: Int, nSpk: Int,
        spkcacheLen: Int, silFramesPerSpk: Int
    ) -> (indices: [Int], isDisabled: [Bool]) {
        let nFramesNoSil = nFrames - silFramesPerSpk
        let maxIndex = 99999

        // Flatten: permute (nFrames, nSpk) → (nSpk, nFrames) → flat
        // flatIndex = speaker * nFrames + frame
        let totalFlat = nSpk * nFrames
        var flatEntries = [(value: Float, flatIdx: Int)]()
        flatEntries.reserveCapacity(totalFlat)
        for s in 0..<nSpk {
            for f in 0..<nFrames {
                flatEntries.append((scores[f * nSpk + s], s * nFrames + f))
            }
        }

        // Sort descending by value, then ascending by flat index for tie-breaking
        flatEntries.sort { a, b in
            if a.value != b.value { return a.value > b.value }
            return a.flatIdx < b.flatIdx
        }

        // Take top spkcacheLen, replace -inf with maxIndex
        var topkIndices = [Int](repeating: maxIndex, count: spkcacheLen)
        let k = min(spkcacheLen, flatEntries.count)
        for i in 0..<k {
            if flatEntries[i].value == -.infinity {
                topkIndices[i] = maxIndex
            } else {
                topkIndices[i] = flatEntries[i].flatIdx
            }
        }

        // Sort to preserve original order (speaker 0 frames first, etc.)
        topkIndices.sort()

        // Map to frame indices and determine disabled mask
        var isDisabled = [Bool](repeating: false, count: spkcacheLen)
        for i in 0..<spkcacheLen {
            if topkIndices[i] == maxIndex {
                isDisabled[i] = true
            }
            topkIndices[i] = topkIndices[i] % nFrames
            if topkIndices[i] >= nFramesNoSil {
                isDisabled[i] = true
            }
            if isDisabled[i] {
                topkIndices[i] = 0  // placeholder for gather
            }
        }

        return (topkIndices, isDisabled)
    }

    /// Gather embeddings and predictions at selected frame indices.
    ///
    /// For disabled frames (silence slots or insufficient data), uses the
    /// mean silence embedding and zero predictions.
    ///
    /// Matches NeMo's `_gather_spkcache_and_preds`.
    static func gatherSpkcacheAndPreds(
        embSeq: [Float], preds: [Float],
        topkIndices: [Int], isDisabled: [Bool],
        meanSilEmb: [Float],
        spkcacheLen: Int, embDim: Int, nSpk: Int
    ) -> Result {
        var outEmb = [Float](repeating: 0, count: spkcacheLen * embDim)
        var outPreds = [Float](repeating: 0, count: spkcacheLen * nSpk)

        for i in 0..<spkcacheLen {
            if isDisabled[i] {
                // Use mean silence embedding, predictions stay zero
                let dstBase = i * embDim
                for d in 0..<embDim {
                    outEmb[dstBase + d] = meanSilEmb[d]
                }
            } else {
                let f = topkIndices[i]
                let srcEmbBase = f * embDim
                let dstEmbBase = i * embDim
                for d in 0..<embDim {
                    outEmb[dstEmbBase + d] = embSeq[srcEmbBase + d]
                }
                let srcPredBase = f * nSpk
                let dstPredBase = i * nSpk
                for s in 0..<nSpk {
                    outPreds[dstPredBase + s] = preds[srcPredBase + s]
                }
            }
        }

        return Result(spkcache: outEmb, spkcachePreds: outPreds)
    }

    // MARK: - Silence Profile

    /// Update running mean silence embedding from new frames.
    ///
    /// A frame is considered silence if the sum of its speaker predictions
    /// is below `silThreshold`.
    ///
    /// Matches NeMo's `_get_silence_profile`.
    static func updateSilenceProfile(
        meanSilEmb: inout [Float],
        nSilFrames: inout Int,
        embSeq: [Float],
        preds: [Float],
        nNewFrames: Int,
        embDim: Int,
        nSpk: Int,
        silThreshold: Float
    ) {
        var silCount = 0
        var silEmbSum = [Float](repeating: 0, count: embDim)

        for f in 0..<nNewFrames {
            var predSum: Float = 0
            let predBase = f * nSpk
            for s in 0..<nSpk {
                predSum += preds[predBase + s]
            }
            if predSum < silThreshold {
                silCount += 1
                let embBase = f * embDim
                for d in 0..<embDim {
                    silEmbSum[d] += embSeq[embBase + d]
                }
            }
        }

        guard silCount > 0 else { return }

        let updatedN = nSilFrames + silCount
        for d in 0..<embDim {
            let oldSum = meanSilEmb[d] * Float(nSilFrames)
            meanSilEmb[d] = (oldSum + silEmbSum[d]) / Float(max(updatedN, 1))
        }
        nSilFrames = updatedN
    }
}
