#!/usr/bin/env python3
"""Run the REAL NeMo AOSC code with PyTorch and print ground truth values.

This uses the actual NeMo SortformerModules methods (copied verbatim),
executed with torch tensors, to produce the one true reference output.
"""
import math
import torch
import numpy as np

torch.set_printoptions(precision=8)
np.set_printoptions(precision=8, suppress=False)


# ─── Minimal SortformerModules with REAL NeMo methods (verbatim from source) ──

class SortformerModules:
    """Minimal mock with only the AOSC methods, copied from NeMo."""

    def __init__(self, n_spk=4, spkcache_len=188, spkcache_sil_frames_per_spk=3,
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

    # ── ALL methods below copied VERBATIM from NeMo sortformer_modules.py ──

    def _get_log_pred_scores(self, preds):
        log_probs = torch.log(torch.clamp(preds, min=self.pred_score_threshold))
        log_1_probs = torch.log(torch.clamp(1.0 - preds, min=self.pred_score_threshold))
        log_1_probs_sum = log_1_probs.sum(dim=2).unsqueeze(-1).expand(-1, -1, self.n_spk)
        scores = log_probs - log_1_probs + log_1_probs_sum - math.log(0.5)
        return scores

    def _disable_low_scores(self, preds, scores, min_pos_scores_per_spk: int):
        is_speech = preds > 0.5
        scores = torch.where(is_speech, scores, torch.tensor(float('-inf')))
        is_pos = scores > 0
        is_nonpos_replace = (~is_pos) * is_speech * (is_pos.sum(dim=1).unsqueeze(1) >= min_pos_scores_per_spk)
        scores = torch.where(is_nonpos_replace, torch.tensor(float('-inf')), scores)
        return scores

    def _boost_topk_scores(self, scores, n_boost_per_spk: int, scale_factor: float = 1.0, offset: float = 0.5):
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

    def _compress_spkcache(self, emb_seq, preds, mean_sil_emb, permute_spk: bool = False):
        batch_size, n_frames, n_spk = preds.shape
        spkcache_len_per_spk = self.spkcache_len // n_spk - self.spkcache_sil_frames_per_spk
        strong_boost_per_spk = math.floor(spkcache_len_per_spk * self.strong_boost_rate)
        weak_boost_per_spk = math.floor(spkcache_len_per_spk * self.weak_boost_rate)
        min_pos_scores_per_spk = math.floor(spkcache_len_per_spk * self.min_pos_scores_rate)

        scores = self._get_log_pred_scores(preds)
        scores = self._disable_low_scores(preds, scores, min_pos_scores_per_spk)

        spk_perm = None  # inference only

        if self.scores_boost_latest > 0:
            scores[:, self.spkcache_len:, :] += self.scores_boost_latest

        # Strong boosting
        scores = self._boost_topk_scores(scores, strong_boost_per_spk, scale_factor=2)
        # Weak boosting
        scores = self._boost_topk_scores(scores, weak_boost_per_spk, scale_factor=1)

        if self.spkcache_sil_frames_per_spk > 0:
            pad = torch.full((batch_size, self.spkcache_sil_frames_per_spk, n_spk), float('inf'), device=scores.device)
            scores = torch.cat([scores, pad], dim=1)

        topk_indices, is_disabled = self._get_topk_indices(scores)
        spkcache, spkcache_preds = self._gather_spkcache_and_preds(
            emb_seq, preds, topk_indices, is_disabled, mean_sil_emb
        )
        return spkcache, spkcache_preds, spk_perm


# ─── Run test cases and print ground truth ────────────────────────────────────

def flat(t):
    """Flatten tensor to list of Python floats."""
    return t.detach().cpu().flatten().tolist()


def print_section(name):
    print(f"\n{'='*60}")
    print(f"NEMO GROUND TRUTH: {name}")
    print(f"{'='*60}")


# ── Test A: _get_log_pred_scores ──
print_section("_get_log_pred_scores (4 frames, 4 speakers)")
mod4 = SortformerModules(n_spk=4, pred_score_threshold=0.25)
preds_a = torch.tensor([[[0.9, 0.1, 0.05, 0.05],
                          [0.1, 0.85, 0.05, 0.05],
                          [0.8, 0.7, 0.05, 0.05],
                          [0.05, 0.05, 0.05, 0.05]]], dtype=torch.float32)
scores_a = mod4._get_log_pred_scores(preds_a)
vals = flat(scores_a)
print("SCORES_A:", " ".join(f"{v:.8f}" for v in vals))


# ── Test B: _disable_low_scores ──
print_section("_disable_low_scores (min_pos=1)")
scores_b = mod4._get_log_pred_scores(preds_a.clone())
scores_b = mod4._disable_low_scores(preds_a, scores_b, min_pos_scores_per_spk=1)
vals = flat(scores_b)
print("DISABLED_B:", " ".join(f"{v:.8f}" for v in vals))


# ── Test C: _boost_topk_scores ──
print_section("_boost_topk_scores (6 frames, 2 speakers)")
mod2 = SortformerModules(n_spk=2)
scores_c = torch.tensor([[[5.0, 1.0],
                           [3.0, 4.0],
                           [1.0, 2.0],
                           [float('-inf'), 6.0],
                           [2.0, float('-inf')],
                           [4.0, 3.0]]], dtype=torch.float32)
scores_c = mod2._boost_topk_scores(scores_c, n_boost_per_spk=2, scale_factor=2.0)
vals = flat(scores_c)
print("BOOSTED_C:", " ".join(f"{v:.8f}" for v in vals))


# ── Test D: _get_topk_indices ──
print_section("_get_topk_indices (8+2 frames, 2 speakers, select 6)")
mod_d = SortformerModules(n_spk=2, spkcache_len=6, spkcache_sil_frames_per_spk=2)
scores_d = torch.tensor([[[5.0, 1.0],
                           [3.0, 4.0],
                           [1.0, 2.0],
                           [float('-inf'), 6.0],
                           [2.0, float('-inf')],
                           [4.0, 3.0],
                           [float('-inf'), float('-inf')],
                           [0.5, 0.5],
                           [float('inf'), float('inf')],
                           [float('inf'), float('inf')]]], dtype=torch.float32)
topk_idx, is_dis = mod_d._get_topk_indices(scores_d)
print("TOPK_IDX:", flat(topk_idx))
print("IS_DISABLED:", [bool(v) for v in flat(is_dis)])


# ── Test E: _get_silence_profile ──
print_section("_get_silence_profile (4 frames)")
mod_sil = SortformerModules(n_spk=2, sil_threshold=0.2)
embs_e = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                         [0.5, 0.5, 0.5, 0.5],
                         [2.0, 3.0, 4.0, 5.0],
                         [0.1, 0.2, 0.3, 0.4]]], dtype=torch.float32)
preds_e = torch.tensor([[[0.8, 0.1],
                          [0.05, 0.05],
                          [0.7, 0.6],
                          [0.05, 0.02]]], dtype=torch.float32)
mean_sil_e = torch.zeros(1, 4, dtype=torch.float32)
n_sil_e = torch.zeros(1, dtype=torch.long)
new_mean, new_n = mod_sil._get_silence_profile(mean_sil_e, n_sil_e, embs_e, preds_e)
print("MEAN_SIL:", flat(new_mean))
print("N_SIL:", int(new_n.item()))


# ── Test F: FULL _compress_spkcache (12 → 8) ──
print_section("_compress_spkcache (12 frames → 8, 2 speakers, emb_dim=4)")
mod_f = SortformerModules(
    n_spk=2, spkcache_len=8, spkcache_sil_frames_per_spk=1,
    pred_score_threshold=0.25, scores_boost_latest=0.05,
    strong_boost_rate=0.75, weak_boost_rate=1.5,
    min_pos_scores_rate=0.5, max_index=99999)

emb_f = torch.zeros(1, 12, 4, dtype=torch.float32)
for f in range(12):
    for d in range(4):
        emb_f[0, f, d] = float(f * 10 + d)

preds_f = torch.tensor([[[0.9, 0.1],
                          [0.85, 0.1],
                          [0.1, 0.9],
                          [0.1, 0.85],
                          [0.8, 0.7],
                          [0.7, 0.8],
                          [0.9, 0.1],
                          [0.1, 0.9],
                          [0.05, 0.05],
                          [0.85, 0.1],
                          [0.1, 0.85],
                          [0.9, 0.1]]], dtype=torch.float32)
mean_sil_f = torch.full((1, 4), 0.5, dtype=torch.float32)

out_emb, out_preds, _ = mod_f._compress_spkcache(emb_f, preds_f, mean_sil_f, permute_spk=False)

emb_vals = flat(out_emb)
pred_vals = flat(out_preds)
print("OUT_EMBS:", " ".join(f"{v:.8f}" for v in emb_vals))
print("OUT_PREDS:", " ".join(f"{v:.8f}" for v in pred_vals))

# Also dump intermediate scores for debug
print("\n--- Intermediate debug ---")
scores_debug = mod_f._get_log_pred_scores(preds_f.clone())
scores_debug = mod_f._disable_low_scores(preds_f, scores_debug, min_pos_scores_per_spk=1)
if mod_f.scores_boost_latest > 0:
    scores_debug[:, mod_f.spkcache_len:, :] += mod_f.scores_boost_latest
print("AFTER_DISABLE+LATEST:")
for f in range(12):
    s0, s1 = scores_debug[0, f, 0].item(), scores_debug[0, f, 1].item()
    print(f"  f{f}: [{s0:.8f}, {s1:.8f}]")

scores_debug2 = scores_debug.clone()
scores_debug2 = mod_f._boost_topk_scores(scores_debug2, 2, scale_factor=2)
print("AFTER_STRONG_BOOST:")
for f in range(12):
    s0, s1 = scores_debug2[0, f, 0].item(), scores_debug2[0, f, 1].item()
    print(f"  f{f}: [{s0:.8f}, {s1:.8f}]")

scores_debug3 = mod_f._boost_topk_scores(scores_debug2, 4, scale_factor=1)
print("AFTER_WEAK_BOOST:")
for f in range(12):
    s0, s1 = scores_debug3[0, f, 0].item(), scores_debug3[0, f, 1].item()
    print(f"  f{f}: [{s0:.8f}, {s1:.8f}]")

print("\n✓ Done — these are the ground truth values from NeMo's actual code with PyTorch.")
