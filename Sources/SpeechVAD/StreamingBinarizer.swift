import Foundation
import AudioCommon

/// Streaming per-speaker binarizer matching NeMo's two-step post-processing.
///
/// NeMo's pipeline (from `vad_utils.py`):
/// 1. `binarization()` — frame-by-frame onset/offset detection + pad_onset/pad_offset
/// 2. `filtering()` — remove short speech segments, fill short silence gaps
///
/// Reference:
///   Paper: Gregory Gelly and Jean-Luc Gauvain. "Minimum Word Error Training of
///          RNN-based Voice Activity Detection", InterSpeech 2015.
///   Implementation: https://github.com/pyannote/pyannote-audio/blob/master/pyannote/audio/utils/signal.py
///
/// All time arithmetic uses Double (64-bit) internally, matching Python's native
/// `float` precision in NeMo. Converts to Float only at the output boundary
/// (DiarizedSegment). This eliminates ~1e-7 drift vs NeMo on long sequences.
///
/// All helper functions (`mergeOverlapSegment`, `filterShortSegments`,
/// `getGapSegments`, `removeSegments`) match NeMo's `vad_utils.py` verbatim
/// in algorithm, translated from PyTorch tensor ops to Swift arrays.
struct StreamingBinarizer {

    // Internal segment type using Double (matches Python float64 precision)
    private struct Seg {
        let start: Double
        let end: Double
    }

    private let numSpeakers: Int
    private let onset: Double
    private let offset: Double
    private let padOnset: Double
    private let padOffset: Double
    private let frameDuration: Double
    private let minDurationOn: Double
    private let minDurationOff: Double
    private let filterSpeechFirst: Double

    /// Per-speaker: whether currently in speech state
    private var inSpeech: [Bool]
    /// Per-speaker: start time of current speech region (Double precision)
    private var speechStart: [Double]
    /// Per-speaker: accumulated raw segments (Double precision, before filtering)
    private var rawSegments: [[Seg]]

    /// Defaults match NeMo's `binarization()` and `filtering()` function defaults.
    /// Callers should pass application-specific values (e.g. DIHARD3, CallHome).
    init(
        numSpeakers: Int,
        onset: Float = 0.5,
        offset: Float = 0.5,
        padOnset: Float = 0.0,
        padOffset: Float = 0.0,
        minDurationOn: Float = 0.0,
        minDurationOff: Float = 0.0,
        filterSpeechFirst: Float = 1.0,
        frameDuration: Float = 0.01
    ) {
        self.numSpeakers = numSpeakers
        self.onset = Double(onset)
        self.offset = Double(offset)
        self.padOnset = Double(padOnset)
        self.padOffset = Double(padOffset)
        self.minDurationOn = Double(minDurationOn)
        self.minDurationOff = Double(minDurationOff)
        self.filterSpeechFirst = Double(filterSpeechFirst)
        self.frameDuration = Double(frameDuration)
        self.inSpeech = [Bool](repeating: false, count: numSpeakers)
        self.speechStart = [Double](repeating: 0, count: numSpeakers)
        self.rawSegments = [[Seg]](repeating: [], count: numSpeakers)
    }

    /// Process new prediction frames (NeMo `binarization()` core loop).
    /// Segments accumulate internally; call `flush()` to get filtered final output.
    mutating func process(probs: [Float], nFrames: Int, baseTime: Float) {
        let baseTimeD = Double(baseTime)

        for f in 0..<nFrames {
            // NeMo: i * frame_length_in_sec (Python float = float64)
            let time = baseTimeD + Double(f) * frameDuration

            for s in 0..<numSpeakers {
                let prob = Double(probs[f * numSpeakers + s])

                // Current frame is speech
                if inSpeech[s] {
                    // Switch from speech to non-speech
                    if prob < offset {
                        let segEnd = time + padOffset
                        let segStart = max(0, speechStart[s] - padOnset)
                        if segEnd > segStart {
                            rawSegments[s].append(Seg(start: segStart, end: segEnd))
                        }
                        speechStart[s] = time
                        inSpeech[s] = false
                    }
                // Current frame is non-speech
                } else {
                    // Switch from non-speech to speech
                    if prob > onset {
                        speechStart[s] = time
                        inSpeech[s] = true
                    }
                }
            }
        }
    }

    /// Finalize: close open segments, then apply NeMo filtering.
    ///
    /// - Parameter endTime: Time of the last frame processed (matching NeMo's
    ///   `i * frame_length_in_sec` after the loop, where `i = len(sequence) - 1`).
    /// - Returns: Filtered, merged segments sorted by start time
    mutating func flush(endTime: Float) -> [DiarizedSegment] {
        let endTimeD = Double(endTime)

        // if it's speech at the end, add final segment
        for s in 0..<numSpeakers {
            if inSpeech[s] {
                let segStart = max(0, speechStart[s] - padOnset)
                let segEnd = endTimeD + padOffset
                rawSegments[s].append(Seg(start: segStart, end: segEnd))
                inSpeech[s] = false
            }
        }

        // Merge the overlapped speech segments due to padding, then apply filtering
        var tagged = [(start: Double, end: Double, speaker: Int)]()
        for s in 0..<numSpeakers {
            var segs = mergeOverlapSegment(rawSegments[s])
            segs = filtering(segs)
            for seg in segs {
                tagged.append((seg.start, seg.end, s))
            }
        }
        rawSegments = [[Seg]](repeating: [], count: numSpeakers)

        tagged.sort { $0.start < $1.start }

        // Convert Double → Float at the output boundary
        return tagged.map {
            DiarizedSegment(startTime: Float($0.start), endTime: Float($0.end), speakerId: $0.speaker)
        }
    }

    /// Which speakers are currently in speech state.
    var activeSpeakers: [Int] {
        (0..<numSpeakers).filter { inSpeech[$0] }
    }

    /// Reset all state for a new audio session.
    mutating func reset() {
        inSpeech = [Bool](repeating: false, count: numSpeakers)
        speechStart = [Double](repeating: 0, count: numSpeakers)
        rawSegments = [[Seg]](repeating: [], count: numSpeakers)
    }

    // MARK: - NeMo vad_utils.py helper functions (all Double precision)

    /// Matches NeMo `merge_overlap_segment()` (vad_utils.py lines 455-475).
    /// Vectorized algorithm: compute merge boundaries, extract group heads/tails.
    private func mergeOverlapSegment(_ segments: [Seg]) -> [Seg] {
        if segments.isEmpty || segments.count == 1 {
            return segments
        }

        let sorted = segments.sorted { $0.start < $1.start }

        // NeMo: merge_boundary = segments[:-1, 1] >= segments[1:, 0]
        var mergeBoundary = [Bool]()
        for i in 0..<(sorted.count - 1) {
            mergeBoundary.append(sorted[i].end >= sorted[i + 1].start)
        }

        // NeMo: head_padded = F.pad(merge_boundary, [1, 0], value=0.0)
        let headPadded = [false] + mergeBoundary
        // NeMo: tail_padded = F.pad(merge_boundary, [0, 1], value=0.0)
        let tailPadded = mergeBoundary + [false]

        // NeMo: head = segments[~head_padded, 0]
        // NeMo: tail = segments[~tail_padded, 1]
        var heads = [Double]()
        var tails = [Double]()
        for i in 0..<sorted.count {
            if !headPadded[i] { heads.append(sorted[i].start) }
            if !tailPadded[i] { tails.append(sorted[i].end) }
        }

        // NeMo: merged = torch.stack((head, tail), dim=1)
        return (0..<heads.count).map { Seg(start: heads[$0], end: tails[$0]) }
    }

    /// Matches NeMo `filter_short_segments()` (vad_utils.py lines 479-487).
    private func filterShortSegments(_ segments: [Seg], threshold: Double) -> [Seg] {
        return segments.filter { $0.end - $0.start >= threshold }
    }

    /// Matches NeMo `get_gap_segments()` (vad_utils.py lines 601-608).
    private func getGapSegments(_ segments: [Seg]) -> [Seg] {
        let sorted = segments.sorted { $0.start < $1.start }
        return (0..<(sorted.count - 1)).map {
            Seg(start: sorted[$0].end, end: sorted[$0 + 1].start)
        }
    }

    /// Matches NeMo `remove_segments()` (vad_utils.py lines 587-597).
    private func removeSegments(_ original: [Seg], removing toBeRemoved: [Seg]) -> [Seg] {
        var result = original
        for y in toBeRemoved {
            result = result.filter { !($0.start == y.start && $0.end == y.end) }
        }
        return result
    }

    /// Matches NeMo `filtering()` (vad_utils.py lines 612-679).
    private func filtering(_ segments: [Seg]) -> [Seg] {
        guard !segments.isEmpty else { return [] }
        var segs = segments

        if filterSpeechFirst == 1.0 {
            // Filter out the shorter speech segments
            if minDurationOn > 0 {
                segs = filterShortSegments(segs, threshold: minDurationOn)
            }
            // Filter out the shorter non-speech segments and return to be as speech segments
            if minDurationOff > 0 {
                // Find non-speech segments
                let nonSpeechSegments = getGapSegments(segs)
                // Find shorter non-speech segments
                let shortNonSpeechSegments = removeSegments(
                    nonSpeechSegments,
                    removing: filterShortSegments(nonSpeechSegments, threshold: minDurationOff))
                // Return shorter non-speech segments to be as speech segments
                segs.append(contentsOf: shortNonSpeechSegments)
                // Merge the overlapped speech segments
                segs = mergeOverlapSegment(segs)
            }
        } else {
            if minDurationOff > 0 {
                // Find non-speech segments
                let nonSpeechSegments = getGapSegments(segs)
                // Find shorter non-speech segments
                let shortNonSpeechSegments = removeSegments(
                    nonSpeechSegments,
                    removing: filterShortSegments(nonSpeechSegments, threshold: minDurationOff))

                segs.append(contentsOf: shortNonSpeechSegments)
                // Merge the overlapped speech segments
                segs = mergeOverlapSegment(segs)
            }
            if minDurationOn > 0 {
                segs = filterShortSegments(segs, threshold: minDurationOn)
            }
        }

        return segs
    }
}
