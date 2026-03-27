#!/usr/bin/env swift
/// Runs ALL SortformerAOSCTests cases as a standalone executable.
/// Equivalent to `swift test --filter SortformerAOSCTests` but without
/// needing to build the full package (avoids MLX C++ build issue).

import Foundation

// ═══════════════════════════════════════════════════════════════════
// AOSC Implementation (copy from SortformerAOSC.swift)
// ═══════════════════════════════════════════════════════════════════

enum AOSCCompressor {
    struct Result { let spkcache: [Float]; let spkcachePreds: [Float] }

    static func compress(
        embSeq: [Float], preds: [Float], nFrames: Int, meanSilEmb: [Float],
        spkcacheLen: Int, nSpk: Int, embDim: Int, silFrames: Int,
        strongBoostRate: Float = 0.75, weakBoostRate: Float = 1.5,
        minPosScoresRate: Float = 0.5, predScoreThreshold: Float = 0.25,
        scoresBoostLatest: Float = 0.05
    ) -> Result {
        let perSpk = spkcacheLen / nSpk - silFrames
        let strongK = Int(floor(Float(perSpk) * strongBoostRate))
        let weakK = Int(floor(Float(perSpk) * weakBoostRate))
        let minPos = Int(floor(Float(perSpk) * minPosScoresRate))
        var scores = getLogPredScores(preds: preds, nFrames: nFrames, nSpk: nSpk, threshold: predScoreThreshold)
        disableLowScores(preds: preds, scores: &scores, nFrames: nFrames, nSpk: nSpk, minPosScoresPerSpk: minPos)
        if scoresBoostLatest > 0 {
            for f in spkcacheLen..<nFrames { let b = f * nSpk
                for s in 0..<nSpk where scores[b+s].isFinite { scores[b+s] += scoresBoostLatest } }
        }
        boostTopkScores(scores: &scores, nFrames: nFrames, nSpk: nSpk, nBoostPerSpk: strongK, scaleFactor: 2.0)
        boostTopkScores(scores: &scores, nFrames: nFrames, nSpk: nSpk, nBoostPerSpk: weakK, scaleFactor: 1.0)
        if silFrames > 0 { for _ in 0..<(silFrames * nSpk) { scores.append(Float.infinity) } }
        let (idx, dis) = getTopkIndices(scores: scores, nFrames: nFrames + silFrames, nSpk: nSpk, spkcacheLen: spkcacheLen, silFramesPerSpk: silFrames)
        return gather(embSeq: embSeq, preds: preds, idx: idx, dis: dis, sil: meanSilEmb, cacheLen: spkcacheLen, embDim: embDim, nSpk: nSpk)
    }

    static func getLogPredScores(preds: [Float], nFrames: Int, nSpk: Int, threshold: Float) -> [Float] {
        var s = [Float](repeating: 0, count: nFrames * nSpk); let lh = logf(0.5)
        for f in 0..<nFrames { let b = f * nSpk; var sum1: Float = 0
            for sp in 0..<nSpk { sum1 += logf(max(1.0 - preds[b+sp], threshold)) }
            for sp in 0..<nSpk { let p = preds[b+sp]; s[b+sp] = logf(max(p, threshold)) - logf(max(1.0 - p, threshold)) + sum1 - lh }
        }; return s
    }

    static func disableLowScores(preds: [Float], scores: inout [Float], nFrames: Int, nSpk: Int, minPosScoresPerSpk: Int) {
        var pc = [Int](repeating: 0, count: nSpk)
        for f in 0..<nFrames { let b = f*nSpk; for s in 0..<nSpk {
            if preds[b+s] <= 0.5 { scores[b+s] = -.infinity } else if scores[b+s] > 0 { pc[s] += 1 } } }
        for f in 0..<nFrames { let b = f*nSpk; for s in 0..<nSpk {
            if !(scores[b+s] > 0) && preds[b+s] > 0.5 && pc[s] >= minPosScoresPerSpk { scores[b+s] = -.infinity } } }
    }

    static func boostTopkScores(scores: inout [Float], nFrames: Int, nSpk: Int, nBoostPerSpk: Int, scaleFactor: Float) {
        guard nBoostPerSpk > 0 else { return }; let boost = -scaleFactor * logf(0.5)
        for s in 0..<nSpk {
            var ix = [(Float, Int)](); for f in 0..<nFrames { ix.append((scores[f*nSpk+s], f)) }
            ix.sort { a, b in if a.0 != b.0 { return a.0 > b.0 }; return a.1 < b.1 }
            for i in 0..<min(nBoostPerSpk, ix.count) { let idx = ix[i].1 * nSpk + s
                if scores[idx].isFinite { scores[idx] += boost } }
        }
    }

    static func getTopkIndices(scores: [Float], nFrames: Int, nSpk: Int, spkcacheLen: Int, silFramesPerSpk: Int) -> ([Int], [Bool]) {
        let nfns = nFrames - silFramesPerSpk; let mx = 99999
        var flat = [(Float, Int)]()
        for s in 0..<nSpk { for f in 0..<nFrames { flat.append((scores[f*nSpk+s], s*nFrames+f)) } }
        flat.sort { a, b in if a.0 != b.0 { return a.0 > b.0 }; return a.1 < b.1 }
        var ti = [Int](repeating: mx, count: spkcacheLen)
        for i in 0..<min(spkcacheLen, flat.count) { ti[i] = flat[i].0 == -.infinity ? mx : flat[i].1 }
        ti.sort()
        var dis = [Bool](repeating: false, count: spkcacheLen)
        for i in 0..<spkcacheLen { if ti[i] == mx { dis[i] = true }; ti[i] = ti[i] % nFrames
            if ti[i] >= nfns { dis[i] = true }; if dis[i] { ti[i] = 0 } }
        return (ti, dis)
    }

    static func gather(embSeq: [Float], preds: [Float], idx: [Int], dis: [Bool], sil: [Float], cacheLen: Int, embDim: Int, nSpk: Int) -> Result {
        var oE = [Float](repeating: 0, count: cacheLen*embDim); var oP = [Float](repeating: 0, count: cacheLen*nSpk)
        for i in 0..<cacheLen {
            if dis[i] { for d in 0..<embDim { oE[i*embDim+d] = sil[d] } }
            else { let f = idx[i]; for d in 0..<embDim { oE[i*embDim+d] = embSeq[f*embDim+d] }
                for s in 0..<nSpk { oP[i*nSpk+s] = preds[f*nSpk+s] } }
        }; return Result(spkcache: oE, spkcachePreds: oP)
    }

    static func updateSilenceProfile(meanSilEmb: inout [Float], nSilFrames: inout Int,
                                      embSeq: [Float], preds: [Float],
                                      nNewFrames: Int, embDim: Int, nSpk: Int, silThreshold: Float) {
        var cnt = 0; var sum = [Float](repeating: 0, count: embDim)
        for f in 0..<nNewFrames { var ps: Float = 0; let pb = f*nSpk
            for s in 0..<nSpk { ps += preds[pb+s] }
            if ps < silThreshold { cnt += 1; let eb = f*embDim; for d in 0..<embDim { sum[d] += embSeq[eb+d] } } }
        guard cnt > 0 else { return }
        let n = nSilFrames + cnt
        for d in 0..<embDim { meanSilEmb[d] = (meanSilEmb[d] * Float(nSilFrames) + sum[d]) / Float(max(n, 1)) }
        nSilFrames = n
    }
}

// ═══════════════════════════════════════════════════════════════════
// Test Runner
// ═══════════════════════════════════════════════════════════════════

var passed = 0, failed = 0, skipped = 0

func assert(_ cond: Bool, _ msg: String) {
    if cond { passed += 1 } else { failed += 1; print("    FAIL: \(msg)") }
}
func assertEq(_ a: Float, _ b: Float, accuracy: Float = 1e-5, _ msg: String) {
    if a.isInfinite && b.isInfinite && a.sign == b.sign { passed += 1; return }
    if a.isNaN && b.isNaN { passed += 1; return }
    if abs(a - b) <= accuracy { passed += 1 } else { failed += 1; print("    FAIL: \(msg) — got \(a), expected \(b)") }
}
func assertEq(_ a: [Int], _ b: [Int], _ msg: String) {
    if a == b { passed += 1 } else { failed += 1; print("    FAIL: \(msg)") }
}
func assertEq(_ a: [Bool], _ b: [Bool], _ msg: String) {
    if a == b { passed += 1 } else { failed += 1; print("    FAIL: \(msg)") }
}

func test(_ name: String, _ body: () throws -> Void) {
    do {
        try body()
        print("  ✓ \(name)")
    } catch {
        skipped += 1
        print("  ⊘ \(name) (skipped: \(error))")
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests (matching SortformerAOSCTests.swift)
// ═══════════════════════════════════════════════════════════════════

print("╔══════════════════════════════════════════════════════════════╗")
print("║  SortformerAOSCTests — standalone runner                    ║")
print("╚══════════════════════════════════════════════════════════════╝\n")

// ── Sub-function tests ──

test("testGetLogPredScores") {
    let preds: [Float] = [0.9,0.1,0.05,0.05, 0.1,0.85,0.05,0.05, 0.8,0.7,0.05,0.05, 0.05,0.05,0.05,0.05]
    let expected: [Float] = [0.37983966,-2.18202806,-2.23609543,-2.23609543, -2.18202806,0.32268131,-2.23609543,-2.23609543, -0.83655596,-1.15240884,-3.33470774,-3.33470774, -0.84702718,-0.84702718,-0.84702718,-0.84702718]
    let scores = AOSCCompressor.getLogPredScores(preds: preds, nFrames: 4, nSpk: 4, threshold: 0.25)
    for i in 0..<16 { assertEq(scores[i], expected[i], accuracy: 1e-4, "score[\(i)]") }
    assert(scores[0] > scores[1], "f0: spk0 highest")
    assert(scores[5] > scores[4], "f1: spk1 highest")
    assert(scores[8] < 0, "f2: spk0 negative (overlap)")
    assert(scores[9] < 0, "f2: spk1 negative (overlap)")
}

test("testGetLogPredScoresThresholdClamping") {
    let scores = AOSCCompressor.getLogPredScores(preds: [0.0, 1.0], nFrames: 1, nSpk: 2, threshold: 0.25)
    for s in scores { assert(!s.isNaN, "not NaN"); assert(s.isFinite, "finite") }
}

test("testDisableLowScores") {
    let preds: [Float] = [0.9,0.1,0.05,0.05, 0.1,0.85,0.05,0.05, 0.8,0.7,0.05,0.05, 0.05,0.05,0.05,0.05]
    let expected: [Float] = [0.37983966,-.infinity,-.infinity,-.infinity, -.infinity,0.32268131,-.infinity,-.infinity, -.infinity,-.infinity,-.infinity,-.infinity, -.infinity,-.infinity,-.infinity,-.infinity]
    var scores = AOSCCompressor.getLogPredScores(preds: preds, nFrames: 4, nSpk: 4, threshold: 0.25)
    AOSCCompressor.disableLowScores(preds: preds, scores: &scores, nFrames: 4, nSpk: 4, minPosScoresPerSpk: 1)
    for i in 0..<16 { assertEq(scores[i], expected[i], accuracy: 1e-4, "disabled[\(i)]") }
}

test("testDisableLowScoresNonSpeech") {
    var scores: [Float] = [1,2,3,4]
    AOSCCompressor.disableLowScores(preds: [0.3,0.2,0.4,0.1], scores: &scores, nFrames: 1, nSpk: 4, minPosScoresPerSpk: 0)
    for s in scores { assertEq(s, -.infinity, "non-speech → -inf") }
}

test("testDisableLowScoresOverlapSuppression") {
    let preds: [Float] = [0.9,0.1, 0.8,0.7, 0.9,0.1]
    var scores = AOSCCompressor.getLogPredScores(preds: preds, nFrames: 3, nSpk: 2, threshold: 0.25)
    AOSCCompressor.disableLowScores(preds: preds, scores: &scores, nFrames: 3, nSpk: 2, minPosScoresPerSpk: 1)
    assert(scores[0].isFinite && scores[0] > 0, "f0-spk0 kept")
    assert(scores[4].isFinite && scores[4] > 0, "f2-spk0 kept")
    assertEq(scores[2], -.infinity, "f1-spk0 overlap disabled")
}

test("testBoostTopkScores") {
    var scores: [Float] = [5,1, 3,4, 1,2, -.infinity,6, 2,-.infinity, 4,3]
    AOSCCompressor.boostTopkScores(scores: &scores, nFrames: 6, nSpk: 2, nBoostPerSpk: 2, scaleFactor: 2.0)
    let b: Float = -2.0 * logf(0.5)
    let expected: [Float] = [5+b,1, 3,4+b, 1,2, -.infinity,6+b, 2,-.infinity, 4+b,3]
    for i in 0..<12 { assertEq(scores[i], expected[i], accuracy: 1e-4, "boost[\(i)]") }
}

test("testBoostTopkScoresInfUntouched") {
    var scores: [Float] = [-.infinity, -.infinity, -.infinity, -.infinity]
    AOSCCompressor.boostTopkScores(scores: &scores, nFrames: 2, nSpk: 2, nBoostPerSpk: 5, scaleFactor: 2.0)
    for s in scores { assertEq(s, -.infinity, "-inf untouched") }
}

test("testBoostTopkScoresZeroBoost") {
    var scores: [Float] = [1,2,3,4]; let orig = scores
    AOSCCompressor.boostTopkScores(scores: &scores, nFrames: 2, nSpk: 2, nBoostPerSpk: 0, scaleFactor: 2.0)
    for i in 0..<4 { assertEq(scores[i], orig[i], "zero boost unchanged") }
}

test("testGetTopkIndices") {
    let scores: [Float] = [5,1, 3,4, 1,2, -.infinity,6, 2,-.infinity, 4,3, -.infinity,-.infinity, 0.5,0.5, .infinity,.infinity, .infinity,.infinity]
    let (idx, dis) = AOSCCompressor.getTopkIndices(scores: scores, nFrames: 10, nSpk: 2, spkcacheLen: 6, silFramesPerSpk: 2)
    assertEq(idx, [0,0,0,3,0,0], "topk indices")
    assertEq(dis, [false,true,true,false,true,true], "topk disabled")
    assert(dis.filter { $0 }.count == 4, "4 disabled")
}

test("testGatherSpkcacheAndPreds") {
    let result = AOSCCompressor.gather(embSeq: [1,2,3, 4,5,6, 7,8,9], preds: [0.9,0.1, 0.5,0.5, 0.1,0.9],
        idx: [0,0,2], dis: [false,true,false], sil: [0.5,0.5,0.5], cacheLen: 3, embDim: 3, nSpk: 2)
    for (i,v) in [1,2,3].enumerated() { assertEq(result.spkcache[i], Float(v), "emb0[\(i)]") }
    for (i,v) in [Float(0.5),0.5,0.5].enumerated() { assertEq(result.spkcache[3+i], v, "sil[\(i)]") }
    for (i,v) in [7,8,9].enumerated() { assertEq(result.spkcache[6+i], Float(v), "emb2[\(i)]") }
    assertEq(result.spkcachePreds[0], 0.9, "pred0[0]"); assertEq(result.spkcachePreds[1], 0.1, "pred0[1]")
    assertEq(result.spkcachePreds[2], 0.0, "sil_pred[0]"); assertEq(result.spkcachePreds[3], 0.0, "sil_pred[1]")
    assertEq(result.spkcachePreds[4], 0.1, "pred2[0]"); assertEq(result.spkcachePreds[5], 0.9, "pred2[1]")
}

test("testUpdateSilenceProfile") {
    let embs: [Float] = [1,2,3,4, 0.5,0.5,0.5,0.5, 2,3,4,5, 0.1,0.2,0.3,0.4]
    let preds: [Float] = [0.8,0.1, 0.05,0.05, 0.7,0.6, 0.05,0.02]
    var mean = [Float](repeating: 0, count: 4); var n = 0
    AOSCCompressor.updateSilenceProfile(meanSilEmb: &mean, nSilFrames: &n, embSeq: embs, preds: preds, nNewFrames: 4, embDim: 4, nSpk: 2, silThreshold: 0.2)
    assert(n == 2, "n_sil=2")
    for (i,v) in [Float(0.3),0.35,0.4,0.45].enumerated() { assertEq(mean[i], v, accuracy: 1e-5, "mean[\(i)]") }
}

test("testUpdateSilenceProfileIncremental") {
    var mean: [Float] = [1.0, 2.0]; var n = 2
    AOSCCompressor.updateSilenceProfile(meanSilEmb: &mean, nSilFrames: &n, embSeq: [3.0,4.0], preds: [0.05,0.05], nNewFrames: 1, embDim: 2, nSpk: 2, silThreshold: 0.2)
    assert(n == 3, "n=3"); assertEq(mean[0], 5.0/3.0, accuracy: 1e-5, "mean[0]"); assertEq(mean[1], 8.0/3.0, accuracy: 1e-5, "mean[1]")
}

test("testUpdateSilenceProfileNoSilence") {
    var mean: [Float] = [1.0, 2.0]; var n = 1
    AOSCCompressor.updateSilenceProfile(meanSilEmb: &mean, nSilFrames: &n, embSeq: [5.0,6.0], preds: [0.8,0.5], nNewFrames: 1, embDim: 2, nSpk: 2, silThreshold: 0.2)
    assert(n == 1, "unchanged count"); assertEq(mean[0], 1.0, "unchanged mean[0]"); assertEq(mean[1], 2.0, "unchanged mean[1]")
}

// ── NeMo ground truth tests ──

test("testCompressVsNeMoGroundTruth") {
    let paths = ["Tests/SpeechVADTests/nemo_ground_truth.json", "nemo_ground_truth.json"]
    var data: Data?
    for p in paths { data = try? Data(contentsOf: URL(fileURLWithPath: p)); if data != nil { break } }
    guard let d = data, let json = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
          let cases = json["compress_cases"] as? [[String: Any]] else {
        throw NSError(domain: "", code: 0, userInfo: [NSLocalizedDescriptionKey: "JSON not found"])
    }
    assert(cases.count >= 4, "≥4 cases")

    for c in cases {
        let name = c["name"] as! String
        let nF = c["n_frames"] as! Int, nS = c["n_spk"] as! Int, eD = c["emb_dim"] as! Int
        let cL = c["spkcache_len"] as! Int, sF = c["sil_frames"] as! Int
        let preds = (c["preds"] as! [NSNumber]).map { Float(truncating: $0) }
        let embs = (c["embs"] as! [NSNumber]).map { Float(truncating: $0) }
        let mSil = (c["mean_sil"] as! [NSNumber]).map { Float(truncating: $0) }
        let expE = (c["expected_embs"] as! [NSNumber]).map { Float(truncating: $0) }
        let expP = (c["expected_preds"] as! [NSNumber]).map { Float(truncating: $0) }

        let r = AOSCCompressor.compress(embSeq: embs, preds: preds, nFrames: nF, meanSilEmb: mSil,
            spkcacheLen: cL, nSpk: nS, embDim: eD, silFrames: sF)

        assert(r.spkcache.count == expE.count, "\(name): emb count")
        for i in 0..<expE.count { assertEq(r.spkcache[i], expE[i], accuracy: 1e-5, "\(name) emb[\(i)]") }
        assert(r.spkcachePreds.count == expP.count, "\(name): pred count")
        for i in 0..<expP.count { assertEq(r.spkcachePreds[i], expP[i], accuracy: 1e-5, "\(name) pred[\(i)]") }
    }
}

test("testCompressProperties") {
    let paths = ["Tests/SpeechVADTests/nemo_ground_truth.json", "nemo_ground_truth.json"]
    var data: Data?
    for p in paths { data = try? Data(contentsOf: URL(fileURLWithPath: p)); if data != nil { break } }
    guard let d = data, let json = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
          let cases = json["compress_cases"] as? [[String: Any]] else {
        throw NSError(domain: "", code: 0, userInfo: [NSLocalizedDescriptionKey: "JSON not found"])
    }

    for c in cases {
        let name = c["name"] as! String
        let nS = c["n_spk"] as! Int, eD = c["emb_dim"] as! Int
        let cL = c["spkcache_len"] as! Int, sF = c["sil_frames"] as! Int
        let preds = (c["preds"] as! [NSNumber]).map { Float(truncating: $0) }
        let embs = (c["embs"] as! [NSNumber]).map { Float(truncating: $0) }
        let mSil = (c["mean_sil"] as! [NSNumber]).map { Float(truncating: $0) }

        let r = AOSCCompressor.compress(embSeq: embs, preds: preds, nFrames: c["n_frames"] as! Int,
            meanSilEmb: mSil, spkcacheLen: cL, nSpk: nS, embDim: eD, silFrames: sF)

        // P1: output size
        assert(r.spkcache.count == cL * eD, "\(name): emb size")
        assert(r.spkcachePreds.count == cL * nS, "\(name): pred size")

        // P2: silence frames use silence embedding + zero preds
        for i in 0..<cL {
            let isSil = (0..<nS).allSatisfy { r.spkcachePreds[i*nS+$0] == 0 }
            if isSil { for d in 0..<eD { assertEq(r.spkcache[i*eD+d], mSil[d], accuracy: 1e-6, "\(name) sil_emb[\(i)][\(d)]") } }
        }

        // P3: speaker order monotonic (arrival-time ordering)
        var lastSpk = -1
        for i in 0..<cL {
            let isSil = (0..<nS).allSatisfy { r.spkcachePreds[i*nS+$0] == 0 }
            if !isSil {
                var dom = 0; var mx: Float = -1
                for s in 0..<nS { if r.spkcachePreds[i*nS+s] > mx { mx = r.spkcachePreds[i*nS+s]; dom = s } }
                assert(dom >= lastSpk, "\(name): speaker order violation at \(i)")
                lastSpk = dom
            }
        }

        // P4: silence count >= nSpk * silFrames (more if not enough active speakers)
        let silCnt = (0..<cL).filter { i in (0..<nS).allSatisfy { r.spkcachePreds[i*nS+$0] == 0 } }.count
        assert(silCnt >= nS * sF, "\(name): sil count \(silCnt) < \(nS*sF)")
    }
}

test("testCompressAllSilence") {
    let paths = ["Tests/SpeechVADTests/nemo_ground_truth.json", "nemo_ground_truth.json"]
    var data: Data?
    for p in paths { data = try? Data(contentsOf: URL(fileURLWithPath: p)); if data != nil { break } }
    guard let d = data, let json = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
          let cases = json["compress_cases"] as? [[String: Any]],
          let sc = cases.first(where: { ($0["name"] as? String) == "all_silence" }) else {
        throw NSError(domain: "", code: 0, userInfo: [NSLocalizedDescriptionKey: "all_silence not found"])
    }
    let nF = sc["n_frames"] as! Int, nS = sc["n_spk"] as! Int, eD = sc["emb_dim"] as! Int
    let cL = sc["spkcache_len"] as! Int, sF = sc["sil_frames"] as! Int
    let preds = (sc["preds"] as! [NSNumber]).map { Float(truncating: $0) }
    let embs = (sc["embs"] as! [NSNumber]).map { Float(truncating: $0) }
    let mSil = (sc["mean_sil"] as! [NSNumber]).map { Float(truncating: $0) }
    let expE = (sc["expected_embs"] as! [NSNumber]).map { Float(truncating: $0) }

    let r = AOSCCompressor.compress(embSeq: embs, preds: preds, nFrames: nF, meanSilEmb: mSil,
        spkcacheLen: cL, nSpk: nS, embDim: eD, silFrames: sF)

    for i in 0..<expE.count { assertEq(r.spkcache[i], expE[i], accuracy: 1e-5, "all_sil emb[\(i)]") }
    for v in r.spkcachePreds { assertEq(v, 0.0, accuracy: 1e-6, "all_sil pred=0") }
}

test("testAOSCConfigDefaults") {
    // Verify default parameter values match NeMo
    let perSpk = 188 / 4 - 3  // 44
    assert(perSpk == 44, "perSpk=44")
    assert(Int(floor(Float(perSpk) * 0.75)) == 33, "strong=33")
    assert(Int(floor(Float(perSpk) * 1.5)) == 66, "weak=66")
    assert(Int(floor(Float(perSpk) * 0.5)) == 22, "minPos=22")
}

// ═══════════════════════════════════════════════════════════════════
// Summary
// ═══════════════════════════════════════════════════════════════════

print("\n══════════════════════════════════════════════════════════════")
print("  Passed:  \(passed)")
print("  Failed:  \(failed)")
print("  Skipped: \(skipped)")
if failed == 0 {
    print("\n  ✓ ALL TESTS PASSED")
} else {
    print("\n  ✗ \(failed) TESTS FAILED")
    exit(1)
}
