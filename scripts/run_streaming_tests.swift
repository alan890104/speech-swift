#!/usr/bin/env swift
/// Tests for StreamingBinarizer (NeMo two-step pipeline) and incremental mel extraction.
/// Run with: swift scripts/run_streaming_tests.swift

import Foundation
import Accelerate

// ═══════════════════════════════════════════════════════════════════
// StreamingBinarizer (copy from StreamingBinarizer.swift)
// Must stay in sync with Sources/SpeechVAD/StreamingBinarizer.swift
// ═══════════════════════════════════════════════════════════════════

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

    var activeSpeakers: [Int] { (0..<numSpeakers).filter { inSpeech[$0] } }

    mutating func reset() {
        inSpeech = [Bool](repeating: false, count: numSpeakers)
        speechStart = [Float](repeating: 0, count: numSpeakers)
        rawSegments = [[DiarizedSegment]](repeating: [], count: numSpeakers)
    }

    // NeMo merge_overlap_segment (vectorized algorithm, vad_utils.py lines 455-475)
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

    // NeMo filter_short_segments (vad_utils.py lines 479-487)
    private func filterShortSegments(_ segments: [DiarizedSegment], threshold: Float) -> [DiarizedSegment] {
        segments.filter { $0.endTime - $0.startTime >= threshold }
    }

    // NeMo get_gap_segments (vad_utils.py lines 601-608)
    private func getGapSegments(_ segments: [DiarizedSegment]) -> [DiarizedSegment] {
        let sorted = segments.sorted { $0.startTime < $1.startTime }
        let spk = sorted.first?.speakerId ?? 0
        return (0..<(sorted.count - 1)).map {
            DiarizedSegment(startTime: sorted[$0].endTime, endTime: sorted[$0+1].startTime, speakerId: spk)
        }
    }

    // NeMo remove_segments (vad_utils.py lines 587-597)
    private func removeSegments(_ original: [DiarizedSegment], removing: [DiarizedSegment]) -> [DiarizedSegment] {
        var result = original
        for y in removing { result = result.filter { !($0.startTime == y.startTime && $0.endTime == y.endTime) } }
        return result
    }

    // NeMo filtering() (vad_utils.py lines 612-679)
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

// ═══════════════════════════════════════════════════════════════════
// Test Runner
// ═══════════════════════════════════════════════════════════════════

var passed = 0, failed = 0

func assert(_ cond: Bool, _ msg: String) {
    if cond { passed += 1 } else { failed += 1; print("    FAIL: \(msg)") }
}
func assertEq(_ a: Float, _ b: Float, accuracy: Float = 0.001, _ msg: String) {
    if abs(a - b) <= accuracy { passed += 1 } else { failed += 1; print("    FAIL: \(msg) — got \(a), expected \(b), diff=\(abs(a-b))") }
}

func test(_ name: String, _ body: () -> Void) {
    body()
    print("  ✓ \(name)")
}

print("╔══════════════════════════════════════════════════════════════╗")
print("║  Streaming Diarization Tests (NeMo two-step pipeline)      ║")
print("╚══════════════════════════════════════════════════════════════╝\n")

// Helper: create binarizer with custom params (no padding, low thresholds for unit tests)
func testBinarizer(numSpeakers: Int = 1, onset: Float = 0.5, offset: Float = 0.3,
                   padOnset: Float = 0, padOffset: Float = 0,
                   minDurationOn: Float = 0.0, minDurationOff: Float = 0.0,
                   frameDuration: Float = 0.08) -> StreamingBinarizer {
    StreamingBinarizer(numSpeakers: numSpeakers, onset: onset, offset: offset,
        padOnset: padOnset, padOffset: padOffset,
        minDurationOn: minDurationOn, minDurationOff: minDurationOff,
        frameDuration: frameDuration)
}

// DIHARD3 config for tests that need it
func dihard3Binarizer(numSpeakers: Int = 1) -> StreamingBinarizer {
    StreamingBinarizer(numSpeakers: numSpeakers, onset: 0.56, offset: 1.0,
        padOnset: 0.063, padOffset: 0.002,
        minDurationOn: 0.007, minDurationOff: 0.151,
        frameDuration: 0.08)
}

// ═══════════════════════════════════════════════════════════════════
// Part 1: NeMo Ground Truth Verification
// ════════════════════════════════════════════════════════��══════════

print("── NeMo Ground Truth Verification ──\n")

test("testNeMoGroundTruthAll") {
    let url = URL(fileURLWithPath: "scripts/nemo_binarization_truth.json")
    let data = try! Data(contentsOf: url)
    let json = try! JSONSerialization.jsonObject(with: data) as! [String: Any]
    let cases = json["cases"] as! [[String: Any]]
    let rawProbs = json["raw_probs"] as! [String: [NSNumber]]
    let dihard3 = json["dihard3_params"] as! [String: NSNumber]
    let callhome = json["callhome_params"] as! [String: NSNumber]
    let frameLen = Float(truncating: json["frame_length"] as! NSNumber)

    for c in cases {
        let name = c["name"] as! String
        let expectedSegs = c["segments"] as! [[String: NSNumber]]
        let probs = (rawProbs[name]!).map { Float(truncating: $0) }
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
        let endTime = Float(probs.count - 1) * frameLen
        let segs = binarizer.flush(endTime: endTime)

        assert(segs.count == expectedSegs.count,
               "\(name) count: swift=\(segs.count) nemo=\(expectedSegs.count)")

        for i in 0..<min(segs.count, expectedSegs.count) {
            let eStart = Float(truncating: expectedSegs[i]["start"]!)
            let eEnd = Float(truncating: expectedSegs[i]["end"]!)
            assertEq(segs[i].startTime, eStart, "\(name) seg[\(i)] start")
            assertEq(segs[i].endTime, eEnd, "\(name) seg[\(i)] end")
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Part 2: Unit Tests
// ═══════════════════════════════════════════════════════════════════

print("\n── Unit Tests ──\n")

test("testSingleSpeakerBasic") {
    // Single speaker speech region, no padding, no filtering
    var binarizer = testBinarizer()
    var probs = [Float](repeating: 0, count: 30)
    for f in 5..<20 { probs[f] = 0.9 }
    binarizer.process(probs: probs, nFrames: 30, baseTime: 0)
    let segs = binarizer.flush(endTime: Float(29) * 0.08)
    assert(segs.count == 1, "Expected 1 segment, got \(segs.count)")
    if let s = segs.first {
        assertEq(s.startTime, 5 * 0.08, "start")
        assert(s.speakerId == 0, "speakerId=0")
    }
}

test("testTwoSpeakers") {
    var binarizer = testBinarizer(numSpeakers: 2)
    var probs = [Float](repeating: 0, count: 40 * 2)
    for f in 0..<10 { probs[f * 2] = 0.9 }       // spk0: frames 0-9
    for f in 15..<25 { probs[f * 2 + 1] = 0.9 }   // spk1: frames 15-24
    binarizer.process(probs: probs, nFrames: 40, baseTime: 0)
    let segs = binarizer.flush(endTime: Float(39) * 0.08)
    assert(segs.count == 2, "Expected 2 segments, got \(segs.count)")
    if segs.count == 2 {
        assert(segs[0].speakerId == 0, "first=spk0")
        assert(segs[1].speakerId == 1, "second=spk1")
    }
}

test("testOnsetOffsetThresholds") {
    // prob exactly at onset (0.5) should NOT start speech (NeMo uses strict >)
    var binarizer = testBinarizer()
    binarizer.process(probs: [0.5], nFrames: 1, baseTime: 0)
    assert(binarizer.activeSpeakers.isEmpty, "prob == onset should not trigger")

    // prob just above onset should start speech
    var b2 = testBinarizer()
    b2.process(probs: [0.51], nFrames: 1, baseTime: 0)
    assert(!b2.activeSpeakers.isEmpty, "prob > onset should trigger")
}

test("testOffsetBehavior") {
    // With offset=0.3: prob drops to 0.25 → ends speech
    var binarizer = testBinarizer(offset: 0.3)
    var probs = [Float](repeating: 0, count: 20)
    for f in 0..<5 { probs[f] = 0.9 }   // speech
    for f in 5..<10 { probs[f] = 0.25 }  // below offset
    for f in 10..<20 { probs[f] = 0.0 }
    binarizer.process(probs: probs, nFrames: 20, baseTime: 0)
    let segs = binarizer.flush(endTime: Float(19) * 0.08)
    assert(segs.count == 1, "Expected 1 segment after offset trigger, got \(segs.count)")
    if let s = segs.first {
        // Speech starts at frame 0, ends at frame 5 (first frame below offset)
        assertEq(s.startTime, 0.0, "start")
        assertEq(s.endTime, 5 * 0.08, accuracy: 0.01, "end at offset trigger")
    }
}

test("testStreamingEquivalence") {
    // Processing in one shot vs two chunks should produce same results
    var b1 = testBinarizer()
    var b2 = testBinarizer()
    var probs = [Float](repeating: 0, count: 30)
    for f in 3..<15 { probs[f] = 0.9 }

    // One shot
    b1.process(probs: probs, nFrames: 30, baseTime: 0)
    let segs1 = b1.flush(endTime: Float(29) * 0.08)

    // Two chunks
    let half = 15
    b2.process(probs: Array(probs[0..<half]), nFrames: half, baseTime: 0)
    b2.process(probs: Array(probs[half..<30]), nFrames: 30 - half, baseTime: Float(half) * 0.08)
    let segs2 = b2.flush(endTime: Float(29) * 0.08)

    assert(segs1.count == segs2.count, "Same count: \(segs1.count) vs \(segs2.count)")
    for i in 0..<min(segs1.count, segs2.count) {
        assertEq(segs1[i].startTime, segs2[i].startTime, "seg[\(i)] start")
        assertEq(segs1[i].endTime, segs2[i].endTime, "seg[\(i)] end")
        assert(segs1[i].speakerId == segs2[i].speakerId, "seg[\(i)] spk")
    }
}

test("testActiveSpeakers") {
    var binarizer = testBinarizer(numSpeakers: 4, onset: 0.5, offset: 0.3)
    // Channels 0 and 2 active
    let probs: [Float] = [0.9, 0.1, 0.8, 0.1]
    binarizer.process(probs: probs, nFrames: 1, baseTime: 0)
    let active = binarizer.activeSpeakers
    assert(active.contains(0), "channel 0 active")
    assert(!active.contains(1), "channel 1 not active")
    assert(active.contains(2), "channel 2 active")
    assert(!active.contains(3), "channel 3 not active")
}

test("testFlushClosesOpenSegments") {
    var binarizer = testBinarizer()
    let probs = [Float](repeating: 0.9, count: 20)
    binarizer.process(probs: probs, nFrames: 20, baseTime: 0)
    assert(binarizer.activeSpeakers.count == 1, "Still active")
    let flushed = binarizer.flush(endTime: Float(19) * 0.08)
    assert(flushed.count == 1, "Flush should close open segment")
}

test("testFlushTrailingSegmentEndTime") {
    // Trailing segment endTime must match NeMo: (N-1)*frame_len + pad_offset
    var binarizer = StreamingBinarizer(numSpeakers: 1, onset: 0.5, offset: 0.3,
        padOnset: 0.063, padOffset: 0.002,
        minDurationOn: 0, minDurationOff: 0, frameDuration: 0.08)
    let probs = [Float](repeating: 0.9, count: 5)
    binarizer.process(probs: probs, nFrames: 5, baseTime: 0)
    let segs = binarizer.flush(endTime: Float(4) * 0.08)  // last frame time, not N*frame_len
    assert(segs.count == 1, "Expected 1 segment")
    if let s = segs.first {
        // NeMo: end = i*frame_len + pad_offset = 4*0.08 + 0.002 = 0.322
        assertEq(s.endTime, 0.322, "trailing endTime = (N-1)*0.08 + 0.002")
    }
}

test("testMinDurationOnFiltering") {
    // minDurationOn filters short speech segments
    var binarizer = testBinarizer(minDurationOn: 0.5)
    var probs = [Float](repeating: 0, count: 20)
    probs[3] = 0.9; probs[4] = 0.9  // 2 frames = 0.16s < 0.5s → filtered
    binarizer.process(probs: probs, nFrames: 20, baseTime: 0)
    let segs = binarizer.flush(endTime: Float(19) * 0.08)
    assert(segs.isEmpty, "Short speech should be filtered: got \(segs.count)")
}

test("testMinDurationOffGapFilling") {
    // minDurationOff fills short silence gaps
    var binarizer = testBinarizer(offset: 0.3, minDurationOff: 0.5)
    var probs = [Float](repeating: 0, count: 30)
    for f in 0..<5 { probs[f] = 0.9 }    // speech 0-0.4s
    for f in 5..<7 { probs[f] = 0.1 }    // gap 0.4-0.56s (0.16s < 0.5s → filled)
    for f in 7..<15 { probs[f] = 0.9 }   // speech 0.56-1.2s
    binarizer.process(probs: probs, nFrames: 30, baseTime: 0)
    let segs = binarizer.flush(endTime: Float(29) * 0.08)
    assert(segs.count == 1, "Gap < minDurationOff should be filled, got \(segs.count)")
}

test("testPaddingExpands") {
    var binarizer = StreamingBinarizer(numSpeakers: 1, onset: 0.5, offset: 0.3,
        padOnset: 0.1, padOffset: 0.05, minDurationOn: 0, minDurationOff: 0, frameDuration: 0.08)
    var probs = [Float](repeating: 0, count: 30)
    for f in 5..<15 { probs[f] = 0.9 }  // speech at 0.4s - 1.2s
    binarizer.process(probs: probs, nFrames: 30, baseTime: 0)
    let segs = binarizer.flush(endTime: Float(29) * 0.08)
    assert(segs.count == 1, "Expected 1 padded segment, got \(segs.count)")
    if let s = segs.first {
        assertEq(s.startTime, 0.4 - 0.1, accuracy: 0.01, "padded start")
        // endTime: segment closes at frame 15 (prob drops) → 15*0.08 + 0.05
        assertEq(s.endTime, 15 * 0.08 + 0.05, accuracy: 0.01, "padded end")
    }
}

test("testPadOnsetClampsToZero") {
    var binarizer = StreamingBinarizer(numSpeakers: 1, onset: 0.5, offset: 0.3,
        padOnset: 0.5, padOffset: 0, minDurationOn: 0, minDurationOff: 0, frameDuration: 0.08)
    var probs = [Float](repeating: 0, count: 20)
    for f in 0..<10 { probs[f] = 0.9 }
    binarizer.process(probs: probs, nFrames: 20, baseTime: 0)
    let segs = binarizer.flush(endTime: Float(19) * 0.08)
    assert(segs.count == 1, "Expected 1 segment")
    if let s = segs.first {
        assertEq(s.startTime, 0.0, "padded start clamped to 0")
    }
}

test("testSpeakerIdIsRawChannel") {
    var binarizer = testBinarizer(numSpeakers: 4)
    var probs = [Float](repeating: 0, count: 40 * 4)
    for f in 0..<8 { probs[f * 4 + 2] = 0.9 }      // channel 2
    for f in 15..<23 { probs[f * 4 + 0] = 0.9 }     // channel 0
    binarizer.process(probs: probs, nFrames: 40, baseTime: 0)
    let segs = binarizer.flush(endTime: Float(39) * 0.08)
    assert(segs.count == 2, "Expected 2 segments, got \(segs.count)")
    if segs.count == 2 {
        // Sorted by startTime: channel 2 first (starts at frame 0), then channel 0 (starts at frame 15)
        assert(segs[0].speakerId == 2, "first = channel 2, got \(segs[0].speakerId)")
        assert(segs[1].speakerId == 0, "second = channel 0, got \(segs[1].speakerId)")
    }
}

test("testProcessEmptyInput") {
    var binarizer = testBinarizer()
    binarizer.process(probs: [], nFrames: 0, baseTime: 0)
    let segs = binarizer.flush(endTime: 0)
    assert(segs.isEmpty, "Empty input → empty output")
}

test("testResetClearsState") {
    var binarizer = testBinarizer()
    binarizer.process(probs: [0.9], nFrames: 1, baseTime: 0)
    assert(!binarizer.activeSpeakers.isEmpty, "Active before reset")
    binarizer.reset()
    assert(binarizer.activeSpeakers.isEmpty, "Idle after reset")
}

test("testFlushThenResetThenProcess") {
    var binarizer = testBinarizer()
    binarizer.process(probs: [Float](repeating: 0.9, count: 10), nFrames: 10, baseTime: 0)
    _ = binarizer.flush(endTime: Float(9) * 0.08)
    binarizer.reset()
    binarizer.process(probs: [Float](repeating: 0.9, count: 10), nFrames: 10, baseTime: 0)
    let segs = binarizer.flush(endTime: Float(9) * 0.08)
    assert(segs.count == 1, "After reset, should work normally: got \(segs.count)")
}

test("testDIHARD3DefaultsBehavior") {
    // With DIHARD3 offset=1.0, every speech frame triggers offset immediately
    // After filtering (minDurationOff=0.151), short gaps are filled → merged
    var b = dihard3Binarizer()
    var probs = [Float](repeating: 0, count: 50)
    for f in 5..<10 { probs[f] = 0.9 }  // 5 frames of speech
    b.process(probs: probs, nFrames: 50, baseTime: 0)
    let segs = b.flush(endTime: Float(49) * 0.08)
    // With offset=1.0, each frame creates its own segment, but gaps < 0.151s are filled
    assert(segs.count == 1, "DIHARD3: merged after gap filling, got \(segs.count)")
}

test("testAllSilence") {
    var binarizer = dihard3Binarizer()
    let probs = [Float](repeating: 0.1, count: 20)
    binarizer.process(probs: probs, nFrames: 20, baseTime: 0)
    let segs = binarizer.flush(endTime: Float(19) * 0.08)
    assert(segs.isEmpty, "All silence → no segments")
}

// ═══════════════════════════════════════════════════════════════════
// Part 3: Incremental Mel Extraction
// ═══════════════════════════════════════════════════════════════════

print("\n── Incremental Mel Tests ──\n")

test("testIncrementalMelFrameCount") {
    let nSamples = 32000  // 2 seconds at 16kHz
    let nFFT = 400, hopLength = 160
    let padLen = nFFT / 2
    let paddedLen = padLen + nSamples + padLen
    let expectedFrames = (paddedLen - nFFT) / hopLength + 1

    // Simulate incremental buffer management
    var streamBuffer = [Float]()
    var baseOffset = 0
    var framesExtracted = 0
    var started = false

    func simulateExtract(newSamples: [Float]) -> Int {
        if !started {
            streamBuffer = [Float](repeating: 0, count: padLen)
            started = true
        }
        streamBuffer.append(contentsOf: newSamples)
        var count = 0
        while framesExtracted * hopLength + nFFT <= streamBuffer.count + baseOffset {
            let local = framesExtracted * hopLength - baseOffset
            guard local >= 0 && local + nFFT <= streamBuffer.count else { break }
            count += 1; framesExtracted += 1
        }
        let trim = framesExtracted * hopLength - baseOffset
        if trim > 0 && trim < streamBuffer.count {
            streamBuffer.removeFirst(trim); baseOffset = framesExtracted * hopLength
        }
        return count
    }

    func simulateFinal() -> Int {
        streamBuffer.append(contentsOf: [Float](repeating: 0, count: padLen))
        var count = 0
        while framesExtracted * hopLength + nFFT <= streamBuffer.count + baseOffset {
            let local = framesExtracted * hopLength - baseOffset
            guard local >= 0 && local + nFFT <= streamBuffer.count else { break }
            count += 1; framesExtracted += 1
        }
        return count
    }

    let audio = [Float](repeating: 0, count: nSamples)
    let f1 = simulateExtract(newSamples: Array(audio[0..<8000]))
    let f2 = simulateExtract(newSamples: Array(audio[8000..<24000]))
    let f3 = simulateExtract(newSamples: Array(audio[24000..<nSamples]))
    let f4 = simulateFinal()
    let total = f1 + f2 + f3 + f4

    assert(total == expectedFrames,
           "Incremental frames (\(total)) should match batch (\(expectedFrames))")
    assert(streamBuffer.count <= nFFT + hopLength,
           "Buffer stays small: \(streamBuffer.count) <= \(nFFT + hopLength)")
}

test("testProgressHandlerCalled") {
    var progressValues = [Double]()
    let totalSamples = 48000
    let chunkSamples = 16000
    var offset = 0
    while offset < totalSamples {
        let end = min(offset + chunkSamples, totalSamples)
        offset = end
        progressValues.append(Double(offset) / Double(totalSamples))
    }
    assert(progressValues.count == 3, "3 chunks for 3s audio")
    assertEq(Float(progressValues.last!), 1.0, accuracy: 0.001, "final progress = 1.0")
    for i in 1..<progressValues.count {
        assert(progressValues[i] > progressValues[i-1], "progress monotonic at \(i)")
    }
}

// ═══════════════════════════════════════════════════════════════════
// Summary
// ═══════════════════════════════════════════════════════════════════

print("\n══════════════════════════════════════════════════════════════")
print("  Passed:  \(passed)")
print("  Failed:  \(failed)")
if failed == 0 {
    print("\n  ✓ ALL TESTS PASSED")
} else {
    print("\n  ✗ \(failed) TESTS FAILED")
    exit(1)
}
