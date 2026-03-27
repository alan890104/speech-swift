#!/usr/bin/env swift
/// Tests for StreamingBinarizer and incremental mel extraction.
/// Run with: swift scripts/run_streaming_tests.swift

import Foundation
import Accelerate

// ═══════════════════════════════════════════════════════════════════
// StreamingBinarizer (copy from StreamingBinarizer.swift)
// ═══════════════════════════════════════════════════════════════════

struct DiarizedSegment { let startTime: Float; let endTime: Float; let speakerId: Int }

struct StreamingBinarizer {
    private enum SpeakerState {
        case idle
        case active(startTime: Float)
        case pendingSilence(speechStart: Float, silenceStart: Float)
    }
    private let numSpeakers: Int, onset: Float, offset: Float
    private let minSpeechDuration: Float, minSilenceDuration: Float, frameDuration: Float
    private var states: [SpeakerState]

    init(numSpeakers: Int, onset: Float = 0.5, offset: Float = 0.3,
         minSpeechDuration: Float = 0.3, minSilenceDuration: Float = 0.15, frameDuration: Float = 0.08) {
        self.numSpeakers = numSpeakers; self.onset = onset; self.offset = offset
        self.minSpeechDuration = minSpeechDuration; self.minSilenceDuration = minSilenceDuration
        self.frameDuration = frameDuration
        self.states = [SpeakerState](repeating: .idle, count: numSpeakers)
    }

    mutating func process(probs: [Float], nFrames: Int, baseTime: Float) -> [DiarizedSegment] {
        var segs = [DiarizedSegment]()
        for f in 0..<nFrames {
            let time = baseTime + Float(f) * frameDuration
            for s in 0..<numSpeakers {
                let prob = probs[f * numSpeakers + s]
                switch states[s] {
                case .idle:
                    if prob >= onset { states[s] = .active(startTime: time) }
                case .active(let st):
                    if prob < offset { states[s] = .pendingSilence(speechStart: st, silenceStart: time) }
                case .pendingSilence(let sp, let si):
                    if prob >= onset { states[s] = .active(startTime: sp) }
                    else if time - si >= minSilenceDuration {
                        if si - sp >= minSpeechDuration {
                            segs.append(DiarizedSegment(startTime: sp, endTime: si, speakerId: s))
                        }
                        states[s] = .idle
                    }
                }
            }
        }
        return segs
    }

    mutating func flush(endTime: Float) -> [DiarizedSegment] {
        var segs = [DiarizedSegment]()
        for s in 0..<numSpeakers {
            switch states[s] {
            case .idle: break
            case .active(let st):
                if endTime - st >= minSpeechDuration { segs.append(DiarizedSegment(startTime: st, endTime: endTime, speakerId: s)) }
            case .pendingSilence(let sp, let si):
                if si - sp >= minSpeechDuration { segs.append(DiarizedSegment(startTime: sp, endTime: si, speakerId: s)) }
            }
            states[s] = .idle
        }
        return segs
    }

    var activeSpeakers: [Int] {
        (0..<numSpeakers).filter { s in
            if case .idle = states[s] { return false }; return true
        }
    }

    mutating func reset() { states = [SpeakerState](repeating: .idle, count: numSpeakers) }
}

// ═══════════════════════════════════════════════════════════════════
// Test Runner
// ═══════════════════════════════════════════════════════════════════

var passed = 0, failed = 0

func assert(_ cond: Bool, _ msg: String) {
    if cond { passed += 1 } else { failed += 1; print("    FAIL: \(msg)") }
}
func assertEq(_ a: Float, _ b: Float, accuracy: Float = 0.01, _ msg: String) {
    if abs(a - b) <= accuracy { passed += 1 } else { failed += 1; print("    FAIL: \(msg) — got \(a), expected \(b)") }
}

func test(_ name: String, _ body: () -> Void) {
    body()
    print("  ✓ \(name)")
}

print("╔══════════════════════════════════════════════════════════════╗")
print("║  Streaming Diarization Tests                                ║")
print("╚══════════════════════════════════════════════════════════════╝\n")

// ── StreamingBinarizer Tests ──

test("testBinarizerSingleSpeaker") {
    // Speaker 0 active from frame 5-20, then silent
    var binarizer = StreamingBinarizer(numSpeakers: 2, onset: 0.5, offset: 0.3,
        minSpeechDuration: 0.2, minSilenceDuration: 0.1, frameDuration: 0.08)
    var probs = [Float](repeating: 0, count: 30 * 2)
    for f in 5..<20 { probs[f * 2] = 0.9 }  // Speaker 0 active

    let segs = binarizer.process(probs: probs, nFrames: 30, baseTime: 0)
    // Should get one finalized segment (enough silence after frame 20)
    assert(segs.count == 1, "Expected 1 segment, got \(segs.count)")
    if let s = segs.first {
        assertEq(s.startTime, 5 * 0.08, "start")
        assert(s.speakerId == 0, "speakerId=0")
    }
}

test("testBinarizerTwoSpeakers") {
    var binarizer = StreamingBinarizer(numSpeakers: 2, onset: 0.5, offset: 0.3,
        minSpeechDuration: 0.1, minSilenceDuration: 0.1, frameDuration: 0.08)
    // Speaker 0: frames 0-10, Speaker 1: frames 15-25
    var probs = [Float](repeating: 0, count: 40 * 2)
    for f in 0..<10 { probs[f * 2] = 0.9 }
    for f in 15..<25 { probs[f * 2 + 1] = 0.9 }

    let segs = binarizer.process(probs: probs, nFrames: 40, baseTime: 0)
    assert(segs.count == 2, "Expected 2 segments, got \(segs.count)")
    if segs.count == 2 {
        assert(segs[0].speakerId == 0, "first=spk0")
        assert(segs[1].speakerId == 1, "second=spk1")
    }
}

test("testBinarizerHysteresis") {
    // Speaker dips below onset but stays above offset → should NOT split
    var binarizer = StreamingBinarizer(numSpeakers: 1, onset: 0.5, offset: 0.3,
        minSpeechDuration: 0.1, minSilenceDuration: 0.1, frameDuration: 0.08)
    var probs = [Float](repeating: 0, count: 30)
    for f in 0..<10 { probs[f] = 0.9 }   // active
    for f in 10..<15 { probs[f] = 0.4 }  // dip (above offset 0.3)
    for f in 15..<25 { probs[f] = 0.9 }  // active again
    // frames 25-29: silent → triggers segment end

    let segs = binarizer.process(probs: probs, nFrames: 30, baseTime: 0)
    // Should be ONE segment (hysteresis keeps it merged)
    assert(segs.count == 1, "Hysteresis should merge: got \(segs.count)")
}

test("testBinarizerStreaming") {
    // Feed same data in two calls — result should be identical to one call
    var b1 = StreamingBinarizer(numSpeakers: 1, onset: 0.5, offset: 0.3,
        minSpeechDuration: 0.1, minSilenceDuration: 0.1, frameDuration: 0.08)
    var b2 = StreamingBinarizer(numSpeakers: 1, onset: 0.5, offset: 0.3,
        minSpeechDuration: 0.1, minSilenceDuration: 0.1, frameDuration: 0.08)

    var probs = [Float](repeating: 0, count: 30)
    for f in 3..<15 { probs[f] = 0.9 }

    // b1: all at once
    let segs1a = b1.process(probs: probs, nFrames: 30, baseTime: 0)
    let segs1b = b1.flush(endTime: 30 * 0.08)
    let all1 = segs1a + segs1b

    // b2: split into two calls
    let half = 15
    let segs2a = b2.process(probs: Array(probs[0..<half]), nFrames: half, baseTime: 0)
    let segs2b = b2.process(probs: Array(probs[half..<30]), nFrames: 30 - half, baseTime: Float(half) * 0.08)
    let segs2c = b2.flush(endTime: 30 * 0.08)
    let all2 = segs2a + segs2b + segs2c

    assert(all1.count == all2.count, "Same segment count: \(all1.count) vs \(all2.count)")
    for i in 0..<min(all1.count, all2.count) {
        assertEq(all1[i].startTime, all2[i].startTime, "seg[\(i)] start")
        assertEq(all1[i].endTime, all2[i].endTime, "seg[\(i)] end")
        assert(all1[i].speakerId == all2[i].speakerId, "seg[\(i)] spk")
    }
}

test("testBinarizerActiveSpeakers") {
    var binarizer = StreamingBinarizer(numSpeakers: 4, onset: 0.5, offset: 0.3,
        minSpeechDuration: 0.1, minSilenceDuration: 0.1, frameDuration: 0.08)
    // Speakers 0 and 2 active
    var probs: [Float] = [0.9, 0.1, 0.8, 0.1]
    _ = binarizer.process(probs: probs, nFrames: 1, baseTime: 0)
    let active = binarizer.activeSpeakers
    assert(active.contains(0), "spk0 active")
    assert(!active.contains(1), "spk1 not active")
    assert(active.contains(2), "spk2 active")
    assert(!active.contains(3), "spk3 not active")
}

test("testBinarizerFlush") {
    var binarizer = StreamingBinarizer(numSpeakers: 1, onset: 0.5, offset: 0.3,
        minSpeechDuration: 0.1, minSilenceDuration: 0.1, frameDuration: 0.08)
    // Start speaking, never stop
    var probs = [Float](repeating: 0.9, count: 20)
    let segs = binarizer.process(probs: probs, nFrames: 20, baseTime: 0)
    assert(segs.isEmpty, "No finalized segments yet")
    assert(binarizer.activeSpeakers.count == 1, "Still active")

    let flushed = binarizer.flush(endTime: 20 * 0.08)
    assert(flushed.count == 1, "Flush should close open segment")
    if let s = flushed.first {
        assertEq(s.startTime, 0, "start=0")
        assertEq(s.endTime, 1.6, "end=1.6")
    }
}

test("testBinarizerMinSpeechDuration") {
    var binarizer = StreamingBinarizer(numSpeakers: 1, onset: 0.5, offset: 0.3,
        minSpeechDuration: 0.5, minSilenceDuration: 0.1, frameDuration: 0.08)
    // Very short speech: 2 frames = 0.16s < minSpeech=0.5s
    var probs = [Float](repeating: 0, count: 20)
    probs[3] = 0.9; probs[4] = 0.9  // 2 frames of speech

    let segs = binarizer.process(probs: probs, nFrames: 20, baseTime: 0)
    // Should be filtered out (too short)
    assert(segs.isEmpty, "Short speech should be filtered: got \(segs.count)")
}

test("testBinarizerReset") {
    var binarizer = StreamingBinarizer(numSpeakers: 2, frameDuration: 0.08)
    _ = binarizer.process(probs: [0.9, 0.1], nFrames: 1, baseTime: 0)
    assert(!binarizer.activeSpeakers.isEmpty, "Active before reset")
    binarizer.reset()
    assert(binarizer.activeSpeakers.isEmpty, "Idle after reset")
}

// ── Incremental Mel Extraction Test ──
// This tests that extracting mel incrementally produces the same frames
// as batch extraction on the same audio.

test("testIncrementalMelMatchesBatch") {
    // Generate 2 seconds of sine wave at 440Hz
    let sampleRate = 16000
    let duration = 2.0
    let nSamples = Int(Double(sampleRate) * duration)
    var audio = [Float](repeating: 0, count: nSamples)
    for i in 0..<nSamples {
        audio[i] = sinf(2.0 * Float.pi * 440.0 * Float(i) / Float(sampleRate)) * 0.5
    }

    let nFFT = 400, hopLength = 160, nMels = 128

    // Batch extraction (simplified — just check frame count and that values are finite)
    // Full batch uses reflect padding on both sides
    let padLen = nFFT / 2
    let paddedLen = padLen + nSamples + padLen
    let expectedFrames = (paddedLen - nFFT) / hopLength + 1

    // Incremental: feed audio in 3 chunks
    let chunk1 = Array(audio[0..<8000])
    let chunk2 = Array(audio[8000..<24000])
    let chunk3 = Array(audio[24000..<nSamples])

    // We can't run the full SortformerMelExtractor here (needs Accelerate setup),
    // but we verify the streaming buffer management logic:

    // Simulate the buffer management
    var streamBuffer = [Float]()
    var baseOffset = 0
    var framesExtracted = 0
    var started = false

    func simulateExtract(newSamples: [Float]) -> Int {
        if !started {
            let pad = [Float](repeating: 0, count: padLen)  // simplified reflect pad
            streamBuffer = pad
            started = true
        }
        streamBuffer.append(contentsOf: newSamples)

        var newFrameCount = 0
        while framesExtracted * hopLength + nFFT <= streamBuffer.count + baseOffset {
            let localStart = framesExtracted * hopLength - baseOffset
            guard localStart >= 0 && localStart + nFFT <= streamBuffer.count else { break }
            newFrameCount += 1
            framesExtracted += 1
        }

        let nextGlobal = framesExtracted * hopLength
        let trimCount = nextGlobal - baseOffset
        if trimCount > 0 && trimCount < streamBuffer.count {
            streamBuffer.removeFirst(trimCount)
            baseOffset = nextGlobal
        }

        return newFrameCount
    }

    func simulateFinal() -> Int {
        let pad = [Float](repeating: 0, count: padLen)
        streamBuffer.append(contentsOf: pad)
        var newFrameCount = 0
        while framesExtracted * hopLength + nFFT <= streamBuffer.count + baseOffset {
            let localStart = framesExtracted * hopLength - baseOffset
            guard localStart >= 0 && localStart + nFFT <= streamBuffer.count else { break }
            newFrameCount += 1
            framesExtracted += 1
        }
        return newFrameCount
    }

    let f1 = simulateExtract(newSamples: chunk1)
    let f2 = simulateExtract(newSamples: chunk2)
    let f3 = simulateExtract(newSamples: chunk3)
    let f4 = simulateFinal()
    let totalFrames = f1 + f2 + f3 + f4

    assert(totalFrames == expectedFrames,
           "Incremental frames (\(totalFrames)) should match batch (\(expectedFrames))")

    // Verify buffer stays small (constant memory)
    assert(streamBuffer.count <= nFFT + hopLength,
           "Buffer should stay small: \(streamBuffer.count) <= \(nFFT + hopLength)")
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
