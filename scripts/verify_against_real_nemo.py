#!/usr/bin/env python3
"""Compare our local binarization functions against REAL NeMo vad_utils.py.

Extracts ONLY the needed functions from the real NeMo repo (cloned to /tmp/nemo_audit),
strips @torch.jit.script decorators, and compares outputs on identical inputs.
"""
import sys
import torch
import json
import random

# ═══════════════════════════════════════════════════════════════
# REAL NeMo functions — extracted verbatim, only decorators stripped
# Source: /tmp/nemo_audit/nemo/collections/asr/parts/utils/vad_utils.py
# ═══════════════════════════════════════════════════════════════

# Lines 455-475 (verbatim, decorator stripped)
def nemo_merge_overlap_segment(segments: torch.Tensor) -> torch.Tensor:
    """
    Merged the given overlapped segments.
    For example:
    torch.Tensor([[0, 1.5], [1, 3.5]]) -> torch.Tensor([0, 3.5])
    """
    if (
        segments.shape == torch.Size([0])
        or segments.shape == torch.Size([0, 2])
        or segments.shape == torch.Size([1, 2])
    ):
        return segments

    segments = segments[segments[:, 0].sort()[1]]
    merge_boundary = segments[:-1, 1] >= segments[1:, 0]
    head_padded = torch.nn.functional.pad(merge_boundary, [1, 0], mode='constant', value=0.0)
    head = segments[~head_padded, 0]
    tail_padded = torch.nn.functional.pad(merge_boundary, [0, 1], mode='constant', value=0.0)
    tail = segments[~tail_padded, 1]
    merged = torch.stack((head, tail), dim=1)
    return merged

# Lines 479-487 (verbatim, decorator stripped)
def nemo_filter_short_segments(segments: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Remove segments which duration is smaller than a threshold.
    """
    return segments[segments[:, 1] - segments[:, 0] >= threshold]

# Lines 587-597 (verbatim, decorator stripped)
def nemo_remove_segments(original_segments: torch.Tensor, to_be_removed_segments: torch.Tensor) -> torch.Tensor:
    """
    Remove speech segments list in to_be_removed_segments from original_segments.
    """
    for y in to_be_removed_segments:
        original_segments = original_segments[original_segments.eq(y).all(dim=1).logical_not()]
    return original_segments

# Lines 601-608 (verbatim, decorator stripped)
def nemo_get_gap_segments(segments: torch.Tensor) -> torch.Tensor:
    """
    Get the gap segments.
    """
    segments = segments[segments[:, 0].sort()[1]]
    return torch.column_stack((segments[:-1, 1], segments[1:, 0]))

# Lines 520-583 (verbatim, decorator stripped, calls nemo_merge_overlap_segment)
def nemo_binarization(sequence: torch.Tensor, per_args) -> torch.Tensor:
    frame_length_in_sec = per_args.get('frame_length_in_sec', 0.01)
    onset = per_args.get('onset', 0.5)
    offset = per_args.get('offset', 0.5)
    pad_onset = per_args.get('pad_onset', 0.0)
    pad_offset = per_args.get('pad_offset', 0.0)

    speech = False
    start = 0.0
    i = 0

    speech_segments = torch.empty(0)

    for i in range(0, len(sequence)):
        # Current frame is speech
        if speech:
            # Switch from speech to non-speech
            if sequence[i] < offset:
                if i * frame_length_in_sec + pad_offset > max(0, start - pad_onset):
                    new_seg = torch.tensor(
                        [max(0, start - pad_onset), i * frame_length_in_sec + pad_offset]
                    ).unsqueeze(0)
                    speech_segments = torch.cat((speech_segments, new_seg), 0)

                start = i * frame_length_in_sec
                speech = False

        # Current frame is non-speech
        else:
            # Switch from non-speech to speech
            if sequence[i] > onset:
                start = i * frame_length_in_sec
                speech = True

    # if it's speech at the end, add final segment
    if speech:
        new_seg = torch.tensor([max(0, start - pad_onset), i * frame_length_in_sec + pad_offset]).unsqueeze(0)
        speech_segments = torch.cat((speech_segments, new_seg), 0)

    # Merge the overlapped speech segments due to padding
    speech_segments = nemo_merge_overlap_segment(speech_segments)  # not sorted
    return speech_segments

# Lines 612-679 (verbatim, decorator stripped, calls nemo_* helpers)
def nemo_filtering(speech_segments: torch.Tensor, per_args) -> torch.Tensor:
    if speech_segments.shape == torch.Size([0]):
        return speech_segments

    min_duration_on = per_args.get('min_duration_on', 0.0)
    min_duration_off = per_args.get('min_duration_off', 0.0)
    filter_speech_first = per_args.get('filter_speech_first', 1.0)

    if filter_speech_first == 1.0:
        # Filter out the shorter speech segments
        if min_duration_on > 0.0:
            speech_segments = nemo_filter_short_segments(speech_segments, min_duration_on)
        # Filter out the shorter non-speech segments and return to be as speech segments
        if min_duration_off > 0.0:
            # Find non-speech segments
            non_speech_segments = nemo_get_gap_segments(speech_segments)
            # Find shorter non-speech segments
            short_non_speech_segments = nemo_remove_segments(
                non_speech_segments, nemo_filter_short_segments(non_speech_segments, min_duration_off)
            )
            # Return shorter non-speech segments to be as speech segments
            speech_segments = torch.cat((speech_segments, short_non_speech_segments), 0)

            # Merge the overlapped speech segments
            speech_segments = nemo_merge_overlap_segment(speech_segments)
    else:
        if min_duration_off > 0.0:
            non_speech_segments = nemo_get_gap_segments(speech_segments)
            short_non_speech_segments = nemo_remove_segments(
                non_speech_segments, nemo_filter_short_segments(non_speech_segments, min_duration_off)
            )
            speech_segments = torch.cat((speech_segments, short_non_speech_segments), 0)
            speech_segments = nemo_merge_overlap_segment(speech_segments)
        if min_duration_on > 0.0:
            speech_segments = nemo_filter_short_segments(speech_segments, min_duration_on)

    return speech_segments


# ═══════════════════════════════════════════════════════════════
# Our LOCAL copies (from nemo_binarization_truth.py)
# ═══════════════════════════════════════════════════════════════

def local_merge_overlap_segment(segments):
    if segments.shape[0] == 0:
        return segments
    segments = segments[segments[:, 0].sort()[1]]
    merged = [segments[0]]
    for seg in segments[1:]:
        if seg[0] <= merged[-1][1]:
            merged[-1] = torch.tensor([merged[-1][0], max(merged[-1][1], seg[1])])
        else:
            merged.append(seg)
    return torch.stack(merged)

def local_filter_short_segments(segments, min_duration):
    if segments.shape[0] == 0:
        return segments
    durations = segments[:, 1] - segments[:, 0]
    mask = durations >= min_duration
    return segments[mask]

def local_get_gap_segments(segments):
    segments = segments[segments[:, 0].sort()[1]]
    return torch.column_stack((segments[:-1, 1], segments[1:, 0]))

def local_remove_segments(original_segments, to_be_removed_segments):
    for y in to_be_removed_segments:
        original_segments = original_segments[original_segments.eq(y).all(dim=1).logical_not()]
    return original_segments

def local_binarization(sequence, per_args):
    frame_length_in_sec = per_args.get('frame_length_in_sec', 0.01)
    onset = per_args.get('onset', 0.5)
    offset = per_args.get('offset', 0.5)
    pad_onset = per_args.get('pad_onset', 0.0)
    pad_offset = per_args.get('pad_offset', 0.0)
    speech = False
    start = 0.0
    i = 0
    speech_segments = torch.empty(0)
    for i in range(0, len(sequence)):
        if speech:
            if sequence[i] < offset:
                if i * frame_length_in_sec + pad_offset > max(0, start - pad_onset):
                    new_seg = torch.tensor(
                        [max(0, start - pad_onset), i * frame_length_in_sec + pad_offset]
                    ).unsqueeze(0)
                    speech_segments = torch.cat((speech_segments, new_seg), 0)
                start = i * frame_length_in_sec
                speech = False
        else:
            if sequence[i] > onset:
                start = i * frame_length_in_sec
                speech = True
    if speech:
        new_seg = torch.tensor([max(0, start - pad_onset), i * frame_length_in_sec + pad_offset]).unsqueeze(0)
        speech_segments = torch.cat((speech_segments, new_seg), 0)
    speech_segments = local_merge_overlap_segment(speech_segments)
    return speech_segments

def local_filtering(speech_segments, per_args):
    if speech_segments.shape == torch.Size([0]):
        return speech_segments
    min_duration_on = per_args.get('min_duration_on', 0.0)
    min_duration_off = per_args.get('min_duration_off', 0.0)
    filter_speech_first = per_args.get('filter_speech_first', 1.0)
    if filter_speech_first == 1.0:
        if min_duration_on > 0.0:
            speech_segments = local_filter_short_segments(speech_segments, min_duration_on)
        if min_duration_off > 0.0 and speech_segments.shape[0] > 0:
            non_speech_segments = local_get_gap_segments(speech_segments)
            short_non_speech_segments = local_remove_segments(
                non_speech_segments, local_filter_short_segments(non_speech_segments, min_duration_off)
            )
            if short_non_speech_segments.shape[0] > 0:
                speech_segments = torch.cat((speech_segments, short_non_speech_segments), 0)
                speech_segments = local_merge_overlap_segment(speech_segments)
    else:
        if min_duration_off > 0.0 and speech_segments.shape[0] > 0:
            non_speech_segments = local_get_gap_segments(speech_segments)
            short_non_speech_segments = local_remove_segments(
                non_speech_segments, local_filter_short_segments(non_speech_segments, min_duration_off)
            )
            if short_non_speech_segments.shape[0] > 0:
                speech_segments = torch.cat((speech_segments, short_non_speech_segments), 0)
                speech_segments = local_merge_overlap_segment(speech_segments)
        if min_duration_on > 0.0:
            speech_segments = local_filter_short_segments(speech_segments, min_duration_on)
    return speech_segments


# ═══════════════════════════════════════════════════════════════
# Test cases
# ═══════════════════════════════════════════════════════════════

FRAME_LEN = 0.08
dihard3 = {"onset": 0.56, "offset": 1.0, "pad_onset": 0.063, "pad_offset": 0.002,
           "min_duration_on": 0.007, "min_duration_off": 0.151}
callhome = {"onset": 0.641, "offset": 0.561, "pad_onset": 0.229, "pad_offset": 0.079,
            "min_duration_on": 0.511, "min_duration_off": 0.296}

test_cases = [
    ("single_speaker_dihard3", [0.0]*5 + [0.9]*15 + [0.0]*10, dihard3),
    ("speaker0_dihard3", [0.0]*3 + [0.9]*10 + [0.0]*5 + [0.9]*8 + [0.0]*4, dihard3),
    ("speaker1_dihard3", [0.0]*15 + [0.8]*10 + [0.0]*5, dihard3),
    ("too_short_dihard3", [0.0]*5 + [0.9]*1 + [0.0]*10, dihard3),
    ("short_gap_dihard3", [0.0]*3 + [0.9]*5 + [0.1]*1 + [0.9]*5 + [0.0]*6, dihard3),
    ("varying_confidence", [0.0]*2 + [0.7, 0.9, 0.8, 0.6, 0.9, 0.95, 0.85, 0.7] + [0.0]*5, dihard3),
    ("offset1_behavior", [0.0]*2 + [0.9]*10 + [0.0]*8, dihard3),
    ("offset1_callhome", [0.0]*2 + [0.9]*10 + [0.0]*8, callhome),
    ("all_silence", [0.1]*20, dihard3),
    ("overlap_spk", [0.0]*2 + [0.9]*8 + [0.0]*10, dihard3),
    ("trailing_speech", [0.9]*5, dihard3),
    ("single_trailing_frame", [0.0]*3 + [0.9]*1, dihard3),
    ("trailing_callhome", [0.0]*2 + [0.9]*8, callhome),
]

random.seed(42)
probs_long = [0.6 + random.random() * 0.35 if (10 <= i < 30 or 50 <= i < 80 or 120 <= i < 160)
              else random.random() * 0.2 for i in range(200)]
test_cases.append(("realistic_long", probs_long, dihard3))

# ═══════════════════════════════════════════════════════════════
# Run comparison
# ═══════════════════════════════════════════════════════════════

print("╔══════════════════════════════════════════════════════════════╗")
print("║  LOCAL copy vs REAL NeMo vad_utils.py — output comparison   ║")
print("╚══════════════════════════════════════════════════════════════╝\n")

all_pass = True
total_values = 0

for name, probs, params in test_cases:
    per_args = {**params, "frame_length_in_sec": FRAME_LEN}
    seq = torch.tensor(probs, dtype=torch.float32)

    nemo_segs = nemo_binarization(seq, per_args)
    nemo_segs = nemo_filtering(nemo_segs, per_args)
    local_segs = local_binarization(seq, per_args)
    local_segs = local_filtering(local_segs, per_args)

    match = True
    if nemo_segs.shape == torch.Size([0]) and local_segs.shape == torch.Size([0]):
        n = 0
    elif nemo_segs.shape != local_segs.shape:
        match = False
        n = -1
    else:
        nemo_sorted = nemo_segs[nemo_segs[:, 0].sort()[1]]
        local_sorted = local_segs[local_segs[:, 0].sort()[1]]
        diff = torch.abs(nemo_sorted - local_sorted).max().item()
        n = nemo_segs.shape[0]
        total_values += n * 2
        if diff > 1e-6:
            match = False

    if match:
        print(f"  ✓ {name:35s} {n} segments — IDENTICAL")
    else:
        all_pass = False
        print(f"  ✗ {name:35s} MISMATCH!")
        print(f"    REAL NeMo: {nemo_segs}")
        print(f"    LOCAL:     {local_segs}")

print(f"\n══════════════════════════════════════════════════════════════")
print(f"  Compared {total_values} values across {len(test_cases)} test cases")
if all_pass:
    print(f"\n  ✓ ALL OUTPUTS IDENTICAL — local copy matches real NeMo exactly")
else:
    print(f"\n  ✗ SOME OUTPUTS DIFFER")
    sys.exit(1)
