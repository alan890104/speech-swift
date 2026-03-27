#!/usr/bin/env swift
/// Verify StreamingBinarizer against NeMo binarization+filtering ground truth.
import Foundation

// ── StreamingBinarizer (copy from StreamingBinarizer.swift) ──

struct DiarizedSegment { let startTime: Float; let endTime: Float; let speakerId: Int }

struct StreamingBinarizer {
    private let numSpeakers: Int, onset: Float, offset: Float
    private let padOnset: Float, padOffset: Float, frameDuration: Float
    private let minDurationOn: Float, minDurationOff: Float
    private var inSpeech: [Bool]
    private var speechStart: [Float]
    private var rawSegments: [[DiarizedSegment]]

    init(numSpeakers: Int, onset: Float = 0.56, offset: Float = 1.0,
         padOnset: Float = 0.063, padOffset: Float = 0.002,
         minDurationOn: Float = 0.007, minDurationOff: Float = 0.151, frameDuration: Float = 0.08) {
        self.numSpeakers = numSpeakers; self.onset = onset; self.offset = offset
        self.padOnset = padOnset; self.padOffset = padOffset
        self.minDurationOn = minDurationOn; self.minDurationOff = minDurationOff
        self.frameDuration = frameDuration
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
                        let segStart = max(0, speechStart[s] - padOnset)
                        let segEnd = time + padOffset
                        if segEnd > segStart {
                            rawSegments[s].append(DiarizedSegment(startTime: segStart, endTime: segEnd, speakerId: s))
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
                let segStart = max(0, speechStart[s] - padOnset)
                let segEnd = endTime + padOffset
                if segEnd > segStart { rawSegments[s].append(DiarizedSegment(startTime: segStart, endTime: segEnd, speakerId: s)) }
                inSpeech[s] = false
            }
        }
        var all = [DiarizedSegment]()
        for s in 0..<numSpeakers {
            var segs = mergeOverlapping(rawSegments[s])
            segs = applyFiltering(segs, speakerId: s)
            all.append(contentsOf: segs)
        }
        rawSegments = [[DiarizedSegment]](repeating: [], count: numSpeakers)
        all.sort { $0.startTime < $1.startTime }
        return all
    }

    private func mergeOverlapping(_ segments: [DiarizedSegment]) -> [DiarizedSegment] {
        guard !segments.isEmpty else { return [] }
        let sorted = segments.sorted { $0.startTime < $1.startTime }
        var merged = [sorted[0]]
        for seg in sorted.dropFirst() {
            if seg.startTime <= merged.last!.endTime {
                let last = merged.removeLast()
                merged.append(DiarizedSegment(startTime: last.startTime, endTime: max(last.endTime, seg.endTime), speakerId: last.speakerId))
            } else { merged.append(seg) }
        }
        return merged
    }

    private func applyFiltering(_ segments: [DiarizedSegment], speakerId: Int) -> [DiarizedSegment] {
        guard !segments.isEmpty else { return [] }
        var segs = segments
        if minDurationOn > 0 { segs = segs.filter { $0.endTime - $0.startTime >= minDurationOn } }
        guard !segs.isEmpty else { return [] }
        if minDurationOff > 0 && segs.count > 1 {
            let sorted = segs.sorted { $0.startTime < $1.startTime }
            var extra = [DiarizedSegment]()
            for i in 0..<(sorted.count - 1) {
                let gapStart = sorted[i].endTime, gapEnd = sorted[i+1].startTime
                if gapEnd - gapStart < minDurationOff {
                    extra.append(DiarizedSegment(startTime: gapStart, endTime: gapEnd, speakerId: speakerId))
                }
            }
            if !extra.isEmpty { segs.append(contentsOf: extra); segs = mergeOverlapping(segs) }
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
    let endTime = Float(probs.count) * frameLen
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
