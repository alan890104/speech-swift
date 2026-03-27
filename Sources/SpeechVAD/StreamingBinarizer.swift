import Foundation
import AudioCommon

/// Streaming per-speaker binarizer matching NeMo's two-step post-processing.
///
/// NeMo's pipeline (from `vad_utils.py`):
/// 1. `binarization()` — frame-by-frame onset/offset detection + pad_onset/pad_offset
/// 2. `filtering()` — remove short speech segments, fill short silence gaps
///
/// This struct accumulates per-frame probabilities and produces segments
/// incrementally. Since NeMo's filtering requires seeing all segments
/// (to find gaps and merge), final filtering is deferred to `flush()`.
///
/// During `process()`, raw binarized segments are accumulated internally.
/// `flush()` applies filtering (short speech removal + short gap filling)
/// and returns the final result.
struct StreamingBinarizer {

    private let numSpeakers: Int
    private let onset: Float
    private let offset: Float
    private let padOnset: Float
    private let padOffset: Float
    private let frameDuration: Float

    /// NeMo filtering parameters
    private let minDurationOn: Float   // short non-speech gap deletion threshold
    private let minDurationOff: Float  // short speech segment deletion threshold

    /// Per-speaker: whether currently in speech state
    private var inSpeech: [Bool]
    /// Per-speaker: start time of current speech region (frame time, before padding)
    private var speechStart: [Float]
    /// Per-speaker: accumulated raw segments (with padding, before filtering)
    private var rawSegments: [[DiarizedSegment]]

    init(
        numSpeakers: Int,
        onset: Float = 0.56,
        offset: Float = 1.0,
        padOnset: Float = 0.063,
        padOffset: Float = 0.002,
        minDurationOn: Float = 0.007,
        minDurationOff: Float = 0.151,
        frameDuration: Float = 0.08
    ) {
        self.numSpeakers = numSpeakers
        self.onset = onset
        self.offset = offset
        self.padOnset = padOnset
        self.padOffset = padOffset
        self.minDurationOn = minDurationOn
        self.minDurationOff = minDurationOff
        self.frameDuration = frameDuration
        self.inSpeech = [Bool](repeating: false, count: numSpeakers)
        self.speechStart = [Float](repeating: 0, count: numSpeakers)
        self.rawSegments = [[DiarizedSegment]](repeating: [], count: numSpeakers)
    }

    /// Process new prediction frames. Segments accumulate internally;
    /// call `flush()` to get filtered final output.
    ///
    /// - Parameters:
    ///   - probs: Flat `[nFrames * numSpeakers]` probabilities in [0, 1]
    ///   - nFrames: Number of new frames
    ///   - baseTime: Timestamp of the first frame
    mutating func process(probs: [Float], nFrames: Int, baseTime: Float) {
        for f in 0..<nFrames {
            let time = baseTime + Float(f) * frameDuration

            for s in 0..<numSpeakers {
                var prob = probs[f * numSpeakers + s]
                if prob > 1.0 || prob < 0.0 {
                    prob = 1.0 / (1.0 + exp(-prob))
                }

                if inSpeech[s] {
                    // NeMo: if sequence[i] < offset → end speech
                    if prob < offset {
                        let segStart = max(0, speechStart[s] - padOnset)
                        let segEnd = time + padOffset
                        if segEnd > segStart {
                            rawSegments[s].append(DiarizedSegment(
                                startTime: segStart, endTime: segEnd, speakerId: s))
                        }
                        speechStart[s] = time
                        inSpeech[s] = false
                    }
                } else {
                    // NeMo: if sequence[i] > onset → start speech
                    if prob > onset {
                        speechStart[s] = time
                        inSpeech[s] = true
                    }
                }
            }
        }
    }

    /// Finalize: close open segments, then apply NeMo filtering
    /// (short speech deletion + short gap filling).
    ///
    /// - Parameter endTime: Timestamp of audio end
    /// - Returns: Filtered, merged segments sorted by start time
    mutating func flush(endTime: Float) -> [DiarizedSegment] {
        // Close any open speech segments
        for s in 0..<numSpeakers {
            if inSpeech[s] {
                let segStart = max(0, speechStart[s] - padOnset)
                let segEnd = endTime + padOffset
                if segEnd > segStart {
                    rawSegments[s].append(DiarizedSegment(
                        startTime: segStart, endTime: segEnd, speakerId: s))
                }
                inSpeech[s] = false
            }
        }

        // Apply NeMo filtering per speaker, then merge all
        var allSegments = [DiarizedSegment]()
        for s in 0..<numSpeakers {
            var segs = mergeOverlapping(rawSegments[s])
            segs = applyFiltering(segs, speakerId: s)
            allSegments.append(contentsOf: segs)
        }
        rawSegments = [[DiarizedSegment]](repeating: [], count: numSpeakers)

        allSegments.sort { $0.startTime < $1.startTime }
        return allSegments
    }

    /// Which speakers are currently in speech state.
    var activeSpeakers: [Int] {
        (0..<numSpeakers).filter { inSpeech[$0] }
    }

    /// Reset all state for a new audio session.
    mutating func reset() {
        inSpeech = [Bool](repeating: false, count: numSpeakers)
        speechStart = [Float](repeating: 0, count: numSpeakers)
        rawSegments = [[DiarizedSegment]](repeating: [], count: numSpeakers)
    }

    // MARK: - NeMo filtering (matches vad_utils.py filtering())

    /// Merge overlapping segments (after padding may cause overlaps).
    private func mergeOverlapping(_ segments: [DiarizedSegment]) -> [DiarizedSegment] {
        guard !segments.isEmpty else { return [] }
        let sorted = segments.sorted { $0.startTime < $1.startTime }
        var merged = [sorted[0]]
        for seg in sorted.dropFirst() {
            if seg.startTime <= merged.last!.endTime {
                let last = merged.removeLast()
                merged.append(DiarizedSegment(
                    startTime: last.startTime,
                    endTime: max(last.endTime, seg.endTime),
                    speakerId: last.speakerId))
            } else {
                merged.append(seg)
            }
        }
        return merged
    }

    /// NeMo's filtering: filter_speech_first=1.0 (default).
    /// 1. Remove speech segments shorter than minDurationOn
    /// 2. Fill non-speech gaps shorter than minDurationOff
    private func applyFiltering(_ segments: [DiarizedSegment], speakerId: Int) -> [DiarizedSegment] {
        guard !segments.isEmpty else { return [] }
        var segs = segments

        // Step 1: Remove short speech segments
        if minDurationOn > 0 {
            segs = segs.filter { $0.endTime - $0.startTime >= minDurationOn }
        }
        guard !segs.isEmpty else { return [] }

        // Step 2: Fill short non-speech gaps
        if minDurationOff > 0 && segs.count > 1 {
            // Find gaps
            var gaps = [(start: Float, end: Float)]()
            let sorted = segs.sorted { $0.startTime < $1.startTime }
            for i in 0..<(sorted.count - 1) {
                gaps.append((sorted[i].endTime, sorted[i + 1].startTime))
            }

            // Find short gaps (< minDurationOff) and add them as speech
            var extra = [DiarizedSegment]()
            for gap in gaps {
                if gap.end - gap.start < minDurationOff {
                    extra.append(DiarizedSegment(
                        startTime: gap.start, endTime: gap.end, speakerId: speakerId))
                }
            }

            if !extra.isEmpty {
                segs.append(contentsOf: extra)
                segs = mergeOverlapping(segs)
            }
        }

        return segs
    }
}
