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
    /// Maps raw model channel index → compacted speaker ID (by first appearance).
    /// Channel 2 activates first → speaker 0, channel 0 next → speaker 1, etc.
    private var channelToSpeaker: [Int: Int]
    /// Next speaker ID to assign
    private var nextSpeakerId: Int

    init(
        numSpeakers: Int,
        onset: Float = 0.641,
        offset: Float = 0.561,
        padOnset: Float = 0.229,
        padOffset: Float = 0.079,
        minSpeechDuration: Float = 0.296,
        minSilenceDuration: Float = 0.511,
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
        self.channelToSpeaker = [:]
        self.nextSpeakerId = 0
    }

    /// Get or assign a compacted speaker ID for a raw model channel.
    private mutating func speakerId(forChannel channel: Int) -> Int {
        if let id = channelToSpeaker[channel] { return id }
        let id = nextSpeakerId
        channelToSpeaker[channel] = id
        nextSpeakerId += 1
        return id
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
                        _ = speakerId(forChannel: s)  // register on first activation
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

    /// Which speakers are currently active (speaking right now), using compacted IDs.
    var activeSpeakers: [Int] {
        var result = [Int]()
        for s in 0..<numSpeakers {
            switch states[s] {
            case .active, .pendingSilence:
                result.append(channelToSpeaker[s] ?? s)
            case .idle:
                break
            }
        }
        return result
    }

    /// Apply padOnset/padOffset and map channel to compacted speaker ID.
    private func padded(_ start: Float, _ end: Float, channel: Int) -> DiarizedSegment? {
        let paddedStart = max(0, start - padOnset)
        let paddedEnd = end + padOffset
        let duration = paddedEnd - paddedStart
        guard duration >= minSpeechDuration else { return nil }
        let spkId = channelToSpeaker[channel] ?? channel
        return DiarizedSegment(startTime: paddedStart, endTime: paddedEnd, speakerId: spkId)
    }

    /// Reset all state for a new audio session.
    mutating func reset() {
        states = [SpeakerState](repeating: .idle, count: numSpeakers)
        channelToSpeaker = [:]
        nextSpeakerId = 0
    }
}
