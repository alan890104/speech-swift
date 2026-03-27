import Foundation
import AudioCommon

/// Streaming per-speaker hysteresis binarizer for diarization.
///
/// Converts frame-level speaker probabilities into `DiarizedSegment`s incrementally.
/// Uses the same onset/offset hysteresis as `PowersetDecoder.binarize`, but maintains
/// state across calls so segments can span multiple `process()` invocations.
///
/// Each speaker channel has an independent state machine:
/// - `idle` → prob ≥ onset → `active(startTime)`
/// - `active` → prob < offset → `pending(startTime, silenceStart)`
/// - `pending` → silence ≥ minSilence → emit segment, → `idle`
/// - `pending` → prob ≥ onset → back to `active`
struct StreamingBinarizer {

    /// Per-speaker state.
    private enum SpeakerState {
        case idle
        case active(startTime: Float)
        case pendingSilence(speechStart: Float, silenceStart: Float)
    }

    private let numSpeakers: Int
    private let onset: Float
    private let offset: Float
    private let padOnset: Float
    private let padOffset: Float
    private let minSpeechDuration: Float
    private let minSilenceDuration: Float
    private let frameDuration: Float

    /// Per-speaker state machines.
    private var states: [SpeakerState]

    init(
        numSpeakers: Int,
        onset: Float = 0.56,
        offset: Float = 1.0,
        padOnset: Float = 0.063,
        padOffset: Float = 0.002,
        minSpeechDuration: Float = 0.151,
        minSilenceDuration: Float = 0.007,
        frameDuration: Float = 0.08
    ) {
        self.numSpeakers = numSpeakers
        self.onset = onset
        self.offset = offset
        self.padOnset = padOnset
        self.padOffset = padOffset
        self.minSpeechDuration = minSpeechDuration
        self.minSilenceDuration = minSilenceDuration
        self.frameDuration = frameDuration
        self.states = [SpeakerState](repeating: .idle, count: numSpeakers)
    }

    /// Process new prediction frames and return finalized segments.
    ///
    /// - Parameters:
    ///   - probs: Flat `[nFrames * numSpeakers]` probabilities in [0, 1]
    ///   - nFrames: Number of new frames
    ///   - baseTime: Timestamp of the first frame in this batch
    /// - Returns: Segments that were finalized (speaker stopped talking long enough)
    mutating func process(probs: [Float], nFrames: Int, baseTime: Float) -> [DiarizedSegment] {
        var segments = [DiarizedSegment]()

        for f in 0..<nFrames {
            let time = baseTime + Float(f) * frameDuration

            for s in 0..<numSpeakers {
                var prob = probs[f * numSpeakers + s]
                // Apply sigmoid if raw logits
                if prob > 1.0 || prob < 0.0 {
                    prob = 1.0 / (1.0 + exp(-prob))
                }

                switch states[s] {
                case .idle:
                    if prob >= onset {
                        states[s] = .active(startTime: time)
                    }

                case .active(let startTime):
                    if prob < offset {
                        states[s] = .pendingSilence(speechStart: startTime, silenceStart: time)
                    }

                case .pendingSilence(let speechStart, let silenceStart):
                    if prob >= onset {
                        // False alarm — speech resumed
                        states[s] = .active(startTime: speechStart)
                    } else if time - silenceStart >= minSilenceDuration {
                        // Silence confirmed — emit segment if long enough (with padding)
                        if let seg = padded(speechStart, silenceStart, channel: s) {
                            segments.append(seg)
                        }
                        states[s] = .idle
                    }
                }
            }
        }

        return segments
    }

    /// Flush any open segments at end of audio.
    ///
    /// - Parameter endTime: Timestamp of the audio end
    /// - Returns: Any remaining active or pending segments
    mutating func flush(endTime: Float) -> [DiarizedSegment] {
        var segments = [DiarizedSegment]()

        for s in 0..<numSpeakers {
            switch states[s] {
            case .idle:
                break
            case .active(let startTime):
                if let seg = padded(startTime, endTime, channel: s) {
                    segments.append(seg)
                }
            case .pendingSilence(let speechStart, let silenceStart):
                if let seg = padded(speechStart, silenceStart, channel: s) {
                    segments.append(seg)
                }
            }
            states[s] = .idle
        }

        return segments
    }

    /// Which speakers are currently active (speaking right now).
    /// Returns raw model channel indices (Sortformer channels are arrival-ordered).
    var activeSpeakers: [Int] {
        var result = [Int]()
        for s in 0..<numSpeakers {
            switch states[s] {
            case .active, .pendingSilence:
                result.append(s)
            case .idle:
                break
            }
        }
        return result
    }

    /// Apply padOnset/padOffset to a raw segment and filter by minSpeechDuration.
    /// Uses raw model channel index as speaker ID (Sortformer channels are arrival-ordered).
    private func padded(_ start: Float, _ end: Float, channel: Int) -> DiarizedSegment? {
        let paddedStart = max(0, start - padOnset)
        let paddedEnd = end + padOffset
        let duration = paddedEnd - paddedStart
        guard duration >= minSpeechDuration else { return nil }
        return DiarizedSegment(startTime: paddedStart, endTime: paddedEnd, speakerId: channel)
    }

    /// Reset all state for a new audio session.
    mutating func reset() {
        states = [SpeakerState](repeating: .idle, count: numSpeakers)
    }
}
