#!/usr/bin/env swift
/// Verify StreamingBinarizer against NeMo binarization+filtering ground truth.
/// Must stay in sync with Sources/SpeechVAD/StreamingBinarizer.swift
import Foundation

// ── StreamingBinarizer (copy from StreamingBinarizer.swift) ──

struct DiarizedSegment { let startTime: Float; let endTime: Float; let speakerId: Int }

struct StreamingBinarizer {
    private let numSpeakers: Int, onset: Float, offset: Float
    private let padOnset: Float, padOffset: Float, frameDuration: Float
    private let minDurationOn: Float, minDurationOff: Float, filterSpeechFirst: Float
    private var inSpeech: [Bool]
    private var speechStart: [Float]
    private var rawSegments: [[DiarizedSegment]]

    init(numSpeakers: Int, onset: Float = 0.5, offset: Float = 0.5,
         padOnset: Float = 0.0, padOffset: Float = 0.0,
         minDurationOn: Float = 0.0, minDurationOff: Float = 0.0,
         filterSpeechFirst: Float = 1.0, frameDuration: Float = 0.01) {
        self.numSpeakers = numSpeakers; self.onset = onset; self.offset = offset
        self.padOnset = padOnset; self.padOffset = padOffset
        self.minDurationOn = minDurationOn; self.minDurationOff = minDurationOff
        self.filterSpeechFirst = filterSpeechFirst; self.frameDuration = frameDuration
        self.inSpeech = [Bool](repeating: false, count: numSpeakers)
        self.speechStart = [Float](repeating: 0, count: numSpeakers)
        self.rawSegments = [[DiarizedSegment]](repeating: [], count: numSpeakers)
    }

    mutating func process(probs: [Float], nFrames: Int, baseTime: Float) {
        for f in 0..<nFrames {
            let time = baseTime + Float(f) * frameDuration
            for s in 0..<numSpeakers {
                let prob = probs[f * numSpeakers + s]
                if inSpeech[s] {
                    if prob < offset {
                        if time + padOffset > max(0, speechStart[s] - padOnset) {
                            rawSegments[s].append(DiarizedSegment(
                                startTime: max(0, speechStart[s] - padOnset),
                                endTime: time + padOffset, speakerId: s))
                        }
                        speechStart[s] = time; inSpeech[s] = false
                    }
                } else {
                    if prob > onset { speechStart[s] = time; inSpeech[s] = true }
                }
            }
        }
    }

    mutating func flush(endTime: Float) -> [DiarizedSegment] {
        for s in 0..<numSpeakers {
            if inSpeech[s] {
                rawSegments[s].append(DiarizedSegment(
                    startTime: max(0, speechStart[s] - padOnset),
                    endTime: endTime + padOffset, speakerId: s))
                inSpeech[s] = false
            }
        }
        var all = [DiarizedSegment]()
        for s in 0..<numSpeakers {
            var segs = mergeOverlapSegment(rawSegments[s])
            segs = filtering(segs, speakerId: s)
            all.append(contentsOf: segs)
        }
        rawSegments = [[DiarizedSegment]](repeating: [], count: numSpeakers)
        all.sort { $0.startTime < $1.startTime }
        return all
    }

    // NeMo merge_overlap_segment (vectorized algorithm)
    private func mergeOverlapSegment(_ segments: [DiarizedSegment]) -> [DiarizedSegment] {
        if segments.isEmpty || segments.count == 1 { return segments }
        let sorted = segments.sorted { $0.startTime < $1.startTime }
        let spk = sorted[0].speakerId
        var mergeBoundary = [Bool]()
        for i in 0..<(sorted.count - 1) { mergeBoundary.append(sorted[i].endTime >= sorted[i+1].startTime) }
        let headPadded = [false] + mergeBoundary
        let tailPadded = mergeBoundary + [false]
        var heads = [Float](), tails = [Float]()
        for i in 0..<sorted.count {
            if !headPadded[i] { heads.append(sorted[i].startTime) }
            if !tailPadded[i] { tails.append(sorted[i].endTime) }
        }
        return (0..<heads.count).map { DiarizedSegment(startTime: heads[$0], endTime: tails[$0], speakerId: spk) }
    }

    // NeMo filter_short_segments
    private func filterShortSegments(_ segments: [DiarizedSegment], threshold: Float) -> [DiarizedSegment] {
        segments.filter { $0.endTime - $0.startTime >= threshold }
    }

    // NeMo get_gap_segments
    private func getGapSegments(_ segments: [DiarizedSegment]) -> [DiarizedSegment] {
        let sorted = segments.sorted { $0.startTime < $1.startTime }
        let spk = sorted.first?.speakerId ?? 0
        return (0..<(sorted.count - 1)).map {
            DiarizedSegment(startTime: sorted[$0].endTime, endTime: sorted[$0+1].startTime, speakerId: spk)
        }
    }

    // NeMo remove_segments
    private func removeSegments(_ original: [DiarizedSegment], removing: [DiarizedSegment]) -> [DiarizedSegment] {
        var result = original
        for y in removing { result = result.filter { !($0.startTime == y.startTime && $0.endTime == y.endTime) } }
        return result
    }

    // NeMo filtering()
    private func filtering(_ segments: [DiarizedSegment], speakerId: Int) -> [DiarizedSegment] {
        guard !segments.isEmpty else { return [] }
        var segs = segments
        if filterSpeechFirst == 1.0 {
            if minDurationOn > 0 { segs = filterShortSegments(segs, threshold: minDurationOn) }
            if minDurationOff > 0 {
                let nonSpeech = getGapSegments(segs)
                let shortNonSpeech = removeSegments(nonSpeech, removing: filterShortSegments(nonSpeech, threshold: minDurationOff))
                segs.append(contentsOf: shortNonSpeech)
                segs = mergeOverlapSegment(segs)
            }
        } else {
            if minDurationOff > 0 {
                let nonSpeech = getGapSegments(segs)
                let shortNonSpeech = removeSegments(nonSpeech, removing: filterShortSegments(nonSpeech, threshold: minDurationOff))
                segs.append(contentsOf: shortNonSpeech)
                segs = mergeOverlapSegment(segs)
            }
            if minDurationOn > 0 { segs = filterShortSegments(segs, threshold: minDurationOn) }
        }
        return segs
    }
}

// ── Load ground truth and verify ──

let url = URL(fileURLWithPath: "scripts/nemo_binarization_truth.json")
let data = try! Data(contentsOf: url)
let json = try! JSONSerialization.jsonObject(with: data) as! [String: Any]
let cases = json["cases"] as! [[String: Any]]
let rawProbs = json["raw_probs"] as! [String: [NSNumber]]
let dihard3 = json["dihard3_params"] as! [String: NSNumber]
let callhome = json["callhome_params"] as! [String: NSNumber]
let frameLen = Float(truncating: json["frame_length"] as! NSNumber)

var passed = 0, failed = 0

func check(_ a: Float, _ b: Float, accuracy: Float = 0.001, _ msg: String) {
    if abs(a - b) <= accuracy { passed += 1 }
    else { failed += 1; print("    FAIL: \(msg) — swift=\(a) nemo=\(b) diff=\(abs(a-b))") }
}

print("╔══════════════════════════════════════════════════════════════╗")
print("║  Binarization: Swift vs NeMo PyTorch Ground Truth          ║")
print("╚══════════════════════════════════════════════════════════════╝\n")

for c in cases {
    let name = c["name"] as! String
    let expectedSegs = c["segments"] as! [[String: NSNumber]]
    let probs = (rawProbs[name]!).map { Float(truncating: $0) }

    // Determine which params to use
    let params: [String: NSNumber] = name.contains("callhome") ? callhome : dihard3

    var binarizer = StreamingBinarizer(
        numSpeakers: 1,
        onset: Float(truncating: params["onset"]!),
        offset: Float(truncating: params["offset"]!),
        padOnset: Float(truncating: params["pad_onset"]!),
        padOffset: Float(truncating: params["pad_offset"]!),
        minDurationOn: Float(truncating: params["min_duration_on"]!),
        minDurationOff: Float(truncating: params["min_duration_off"]!),
        frameDuration: frameLen)

    binarizer.process(probs: probs, nFrames: probs.count, baseTime: 0)
    // endTime = last frame time, matching NeMo's `i * frame_length_in_sec` (i = N-1)
    let endTime = Float(probs.count - 1) * frameLen
    let segs = binarizer.flush(endTime: endTime)

    // Compare segment count
    let countMatch = segs.count == expectedSegs.count
    if countMatch { passed += 1 } else {
        failed += 1
        print("    FAIL: \(name) count — swift=\(segs.count) nemo=\(expectedSegs.count)")
    }

    // Compare each segment
    for i in 0..<min(segs.count, expectedSegs.count) {
        let eStart = Float(truncating: expectedSegs[i]["start"]!)
        let eEnd = Float(truncating: expectedSegs[i]["end"]!)
        check(segs[i].startTime, eStart, "\(name) seg[\(i)] start")
        check(segs[i].endTime, eEnd, "\(name) seg[\(i)] end")
    }

    let status = (countMatch && segs.count == expectedSegs.count) ? "✓" : "✗"
    print("  \(status) \(name.padding(toLength: 35, withPad: " ", startingAt: 0)) \(segs.count) segments")
}

print("\n══════════════════════════════════════════════════════════════")
print("  Passed:  \(passed)")
print("  Failed:  \(failed)")
if failed == 0 { print("\n  ✓ ALL VALUES MATCH NeMo ground truth") }
else { print("\n  ✗ SOME VALUES DIFFER"); exit(1) }
