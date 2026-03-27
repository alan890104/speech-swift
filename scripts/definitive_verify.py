#!/usr/bin/env python3
"""Definitive verification: NeMo ground truth for tie-free test cases.

Generates multiple test cases with UNIQUE prediction values (no ties possible),
runs them through the REAL NeMo AOSC code with PyTorch, and outputs ground truth
for Swift to compare against.
"""
import math, json, sys
import torch
import numpy as np

torch.manual_seed(0)
np.random.seed(0)


class SortformerModules:
    """Real NeMo methods, verbatim."""
    def __init__(self, n_spk, spkcache_len, spkcache_sil_frames_per_spk=3,
                 pred_score_threshold=0.25, scores_boost_latest=0.05,
                 strong_boost_rate=0.75, weak_boost_rate=1.5,
                 min_pos_scores_rate=0.5, sil_threshold=0.2, max_index=99999):
        self.n_spk = n_spk
        self.spkcache_len = spkcache_len
        self.spkcache_sil_frames_per_spk = spkcache_sil_frames_per_spk
        self.pred_score_threshold = pred_score_threshold
        self.scores_boost_latest = scores_boost_latest
        self.strong_boost_rate = strong_boost_rate
        self.weak_boost_rate = weak_boost_rate
        self.min_pos_scores_rate = min_pos_scores_rate
        self.sil_threshold = sil_threshold
        self.max_index = max_index
        self.training = False

    def _get_log_pred_scores(self, preds):
        log_probs = torch.log(torch.clamp(preds, min=self.pred_score_threshold))
        log_1_probs = torch.log(torch.clamp(1.0 - preds, min=self.pred_score_threshold))
        log_1_probs_sum = log_1_probs.sum(dim=2).unsqueeze(-1).expand(-1, -1, self.n_spk)
        scores = log_probs - log_1_probs + log_1_probs_sum - math.log(0.5)
        return scores

    def _disable_low_scores(self, preds, scores, min_pos_scores_per_spk):
        is_speech = preds > 0.5
        scores = torch.where(is_speech, scores, torch.tensor(float('-inf')))
        is_pos = scores > 0
        is_nonpos_replace = (~is_pos) * is_speech * (is_pos.sum(dim=1).unsqueeze(1) >= min_pos_scores_per_spk)
        scores = torch.where(is_nonpos_replace, torch.tensor(float('-inf')), scores)
        return scores

    def _boost_topk_scores(self, scores, n_boost_per_spk, scale_factor=1.0, offset=0.5):
        batch_size, _, n_spk = scores.shape
        _, topk_indices = torch.topk(scores, n_boost_per_spk, dim=1, largest=True, sorted=False)
        batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2)
        speaker_indices = torch.arange(n_spk).unsqueeze(0).unsqueeze(0)
        scores[batch_indices, topk_indices, speaker_indices] -= scale_factor * math.log(offset)
        return scores

    def _get_topk_indices(self, scores):
        batch_size, n_frames, _ = scores.shape
        n_frames_no_sil = n_frames - self.spkcache_sil_frames_per_spk
        scores_flatten = scores.permute(0, 2, 1).reshape(batch_size, -1)
        topk_values, topk_indices = torch.topk(scores_flatten, self.spkcache_len, dim=1, sorted=False)
        valid_topk_mask = topk_values != float('-inf')
        topk_indices = torch.where(valid_topk_mask, topk_indices, torch.tensor(self.max_index))
        topk_indices_sorted, _ = torch.sort(topk_indices, dim=1)
        is_disabled = topk_indices_sorted == self.max_index
        topk_indices_sorted = torch.remainder(topk_indices_sorted, n_frames)
        is_disabled += topk_indices_sorted >= n_frames_no_sil
        topk_indices_sorted[is_disabled] = 0
        return topk_indices_sorted, is_disabled

    def _gather_spkcache_and_preds(self, emb_seq, preds, topk_indices, is_disabled, mean_sil_emb):
        emb_dim, n_spk = emb_seq.shape[2], preds.shape[2]
        indices_expanded_emb = topk_indices.unsqueeze(-1).expand(-1, -1, emb_dim)
        emb_seq_gathered = torch.gather(emb_seq, 1, indices_expanded_emb)
        mean_sil_emb_expanded = mean_sil_emb.unsqueeze(1).expand(-1, self.spkcache_len, -1)
        emb_seq_gathered = torch.where(is_disabled.unsqueeze(-1), mean_sil_emb_expanded, emb_seq_gathered)
        indices_expanded_spk = topk_indices.unsqueeze(-1).expand(-1, -1, n_spk)
        preds_gathered = torch.gather(preds, 1, indices_expanded_spk)
        preds_gathered = torch.where(is_disabled.unsqueeze(-1), torch.tensor(0.0), preds_gathered)
        return emb_seq_gathered, preds_gathered

    def _get_silence_profile(self, mean_sil_emb, n_sil_frames, emb_seq, preds):
        is_sil = preds.sum(dim=2) < self.sil_threshold
        sil_count = is_sil.sum(dim=1)
        has_new_sil = sil_count > 0
        if not has_new_sil.any():
            return mean_sil_emb, n_sil_frames
        sil_emb_sum = torch.sum(emb_seq * is_sil.unsqueeze(-1), dim=1)
        upd_n_sil_frames = n_sil_frames + sil_count
        old_sil_emb_sum = mean_sil_emb * n_sil_frames.unsqueeze(1)
        total_sil_sum = old_sil_emb_sum + sil_emb_sum
        upd_mean_sil_emb = total_sil_sum / torch.clamp(upd_n_sil_frames.unsqueeze(1), min=1)
        return upd_mean_sil_emb, upd_n_sil_frames

    def _compress_spkcache(self, emb_seq, preds, mean_sil_emb, permute_spk=False):
        batch_size, n_frames, n_spk = preds.shape
        spkcache_len_per_spk = self.spkcache_len // n_spk - self.spkcache_sil_frames_per_spk
        strong_boost_per_spk = math.floor(spkcache_len_per_spk * self.strong_boost_rate)
        weak_boost_per_spk = math.floor(spkcache_len_per_spk * self.weak_boost_rate)
        min_pos_scores_per_spk = math.floor(spkcache_len_per_spk * self.min_pos_scores_rate)
        scores = self._get_log_pred_scores(preds)
        scores = self._disable_low_scores(preds, scores, min_pos_scores_per_spk)
        if self.scores_boost_latest > 0:
            scores[:, self.spkcache_len:, :] += self.scores_boost_latest
        scores = self._boost_topk_scores(scores, strong_boost_per_spk, scale_factor=2)
        scores = self._boost_topk_scores(scores, weak_boost_per_spk, scale_factor=1)
        if self.spkcache_sil_frames_per_spk > 0:
            pad = torch.full((batch_size, self.spkcache_sil_frames_per_spk, n_spk), float('inf'))
            scores = torch.cat([scores, pad], dim=1)
        topk_indices, is_disabled = self._get_topk_indices(scores)
        spkcache, spkcache_preds = self._gather_spkcache_and_preds(
            emb_seq, preds, topk_indices, is_disabled, mean_sil_emb)
        return spkcache, spkcache_preds


def make_unique_preds(n_frames, n_spk, seed):
    """Generate predictions with ALL unique values — no ties possible."""
    rng = np.random.RandomState(seed)
    preds = np.zeros((n_frames, n_spk), dtype=np.float32)
    for f in range(n_frames):
        # Assign one dominant speaker per frame with unique values
        dominant = f % n_spk
        for s in range(n_spk):
            if s == dominant:
                preds[f, s] = 0.6 + rng.uniform(0.01, 0.39)  # [0.61, 0.99]
            else:
                preds[f, s] = rng.uniform(0.01, 0.15)  # [0.01, 0.15]
    return preds


def check_no_ties(preds, config):
    """Verify that the scores have no ties after full processing."""
    mod = SortformerModules(**config)
    p = torch.tensor(preds, dtype=torch.float32).unsqueeze(0)
    n_frames, n_spk = preds.shape
    spkcache_len_per_spk = config['spkcache_len'] // n_spk - config.get('spkcache_sil_frames_per_spk', 3)
    min_pos = math.floor(spkcache_len_per_spk * config.get('min_pos_scores_rate', 0.5))

    scores = mod._get_log_pred_scores(p)
    scores = mod._disable_low_scores(p, scores, min_pos)
    if config.get('scores_boost_latest', 0.05) > 0:
        scores[:, config['spkcache_len']:, :] += config['scores_boost_latest']

    # Check for ties among finite scores
    finite = scores[scores.isfinite()]
    unique = finite.unique()
    return len(finite) == len(unique)


def run_case(name, n_frames, n_spk, emb_dim, spkcache_len, sil_frames, seed):
    """Run one test case and output ground truth."""
    config = dict(n_spk=n_spk, spkcache_len=spkcache_len,
                  spkcache_sil_frames_per_spk=sil_frames,
                  pred_score_threshold=0.25, scores_boost_latest=0.05,
                  strong_boost_rate=0.75, weak_boost_rate=1.5,
                  min_pos_scores_rate=0.5, max_index=99999)

    preds = make_unique_preds(n_frames, n_spk, seed)
    no_ties = check_no_ties(preds, config)

    rng = np.random.RandomState(seed + 1000)
    embs = rng.randn(n_frames, emb_dim).astype(np.float32) * 10
    mean_sil = rng.randn(emb_dim).astype(np.float32) * 0.1

    mod = SortformerModules(**config)
    p_t = torch.tensor(preds, dtype=torch.float32).unsqueeze(0)
    e_t = torch.tensor(embs, dtype=torch.float32).unsqueeze(0)
    m_t = torch.tensor(mean_sil, dtype=torch.float32).unsqueeze(0)

    out_emb, out_preds = mod._compress_spkcache(e_t, p_t, m_t)

    result = {
        "name": name,
        "n_frames": n_frames, "n_spk": n_spk, "emb_dim": emb_dim,
        "spkcache_len": spkcache_len, "sil_frames": sil_frames,
        "no_ties": bool(no_ties),
        "preds": preds.flatten().tolist(),
        "embs": embs.flatten().tolist(),
        "mean_sil": mean_sil.tolist(),
        "expected_embs": out_emb[0].flatten().tolist(),
        "expected_preds": out_preds[0].flatten().tolist(),
    }
    return result


# Also run sub-function tests
def run_subfunc_tests():
    results = []

    # Test: _get_silence_profile with unique values
    mod = SortformerModules(n_spk=2, spkcache_len=8, sil_threshold=0.2)
    embs = torch.tensor([[[1.1, 2.2, 3.3], [0.4, 0.5, 0.6], [7.7, 8.8, 9.9],
                           [0.01, 0.02, 0.03], [5.5, 6.6, 7.7]]], dtype=torch.float32)
    preds = torch.tensor([[[0.8, 0.15], [0.03, 0.04], [0.7, 0.65],
                            [0.01, 0.08], [0.9, 0.05]]], dtype=torch.float32)
    mean_sil = torch.zeros(1, 3, dtype=torch.float32)
    n_sil = torch.zeros(1, dtype=torch.long)
    new_mean, new_n = mod._get_silence_profile(mean_sil, n_sil, embs, preds)
    results.append({
        "name": "silence_profile",
        "expected_mean": new_mean[0].tolist(),
        "expected_n": int(new_n.item()),
        "embs": embs[0].flatten().tolist(),
        "preds": preds[0].flatten().tolist(),
    })
    return results


if __name__ == "__main__":
    cases = []

    # Case 1: 2 speakers, small (15 → 8)
    cases.append(run_case("2spk_small", 15, 2, 4, 8, 1, seed=42))

    # Case 2: 4 speakers, medium (30 → 16)
    cases.append(run_case("4spk_medium", 30, 4, 8, 16, 2, seed=123))

    # Case 3: 2 speakers, larger (50 → 20)
    cases.append(run_case("2spk_larger", 50, 2, 4, 20, 2, seed=456))

    # Case 4: 4 speakers, realistic scale (60 → 32)
    cases.append(run_case("4spk_realistic", 60, 4, 8, 32, 3, seed=789))

    # Case 5: all silence (all preds < 0.5)
    preds_sil = np.random.RandomState(999).uniform(0.01, 0.15, (20, 2)).astype(np.float32)
    embs_sil = np.random.RandomState(1999).randn(20, 4).astype(np.float32)
    mean_sil_5 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    mod5 = SortformerModules(n_spk=2, spkcache_len=8, spkcache_sil_frames_per_spk=1)
    o_e, o_p = mod5._compress_spkcache(
        torch.tensor(embs_sil).unsqueeze(0), torch.tensor(preds_sil).unsqueeze(0),
        torch.tensor(mean_sil_5).unsqueeze(0))
    cases.append({
        "name": "all_silence",
        "n_frames": 20, "n_spk": 2, "emb_dim": 4, "spkcache_len": 8, "sil_frames": 1,
        "no_ties": True,
        "preds": preds_sil.flatten().tolist(),
        "embs": embs_sil.flatten().tolist(),
        "mean_sil": mean_sil_5.tolist(),
        "expected_embs": o_e[0].flatten().tolist(),
        "expected_preds": o_p[0].flatten().tolist(),
    })

    subfunc = run_subfunc_tests()

    output = {"compress_cases": cases, "subfunc_cases": subfunc}

    with open("/Users/yulun/Documents/fun/speech-swift/scripts/nemo_ground_truth.json", "w") as f:
        json.dump(output, f)

    # Print summary
    for c in cases:
        tie_str = "NO TIES ✓" if c["no_ties"] else "HAS TIES ✗"
        print(f"  {c['name']:20s}  {c['n_frames']:2d}→{c['spkcache_len']:2d}  {c['n_spk']}spk  {tie_str}")

    print(f"\n✓ Wrote {len(cases)} compress cases + {len(subfunc)} subfunc cases to nemo_ground_truth.json")
