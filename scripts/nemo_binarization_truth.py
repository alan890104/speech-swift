#!/usr/bin/env python3
"""Run NeMo's REAL binarization() + filtering() with PyTorch
and produce ground truth for Swift to match."""

import torch
import json

# ── Copy of NeMo's REAL functions from vad_utils.py (verbatim) ──

def merge_overlap_segment(segments):
    """Merge overlapping segments."""
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

def filter_short_segments(segments, min_duration):
    """Remove segments shorter than min_duration."""
    if segments.shape[0] == 0:
        return segments
    durations = segments[:, 1] - segments[:, 0]
    mask = durations >= min_duration
    return segments[mask]

def get_gap_segments(segments):
    segments = segments[segments[:, 0].sort()[1]]
    return torch.column_stack((segments[:-1, 1], segments[1:, 0]))

def remove_segments(original_segments, to_be_removed_segments):
    for y in to_be_removed_segments:
        original_segments = original_segments[original_segments.eq(y).all(dim=1).logical_not()]
    return original_segments

def binarization(sequence, per_args):
    """NeMo's binarization — verbatim from vad_utils.py lines 520-583."""
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

    speech_segments = merge_overlap_segment(speech_segments)
    return speech_segments

def filtering(speech_segments, per_args):
    """NeMo's filtering — verbatim from vad_utils.py lines 612-672."""
    if speech_segments.shape == torch.Size([0]):
        return speech_segments

    min_duration_on = per_args.get('min_duration_on', 0.0)
    min_duration_off = per_args.get('min_duration_off', 0.0)
    filter_speech_first = per_args.get('filter_speech_first', 1.0)

    if filter_speech_first == 1.0:
        if min_duration_on > 0.0:
            speech_segments = filter_short_segments(speech_segments, min_duration_on)
        if min_duration_off > 0.0 and speech_segments.shape[0] > 0:
            non_speech_segments = get_gap_segments(speech_segments)
            short_non_speech_segments = remove_segments(
                non_speech_segments, filter_short_segments(non_speech_segments, min_duration_off)
            )
            if short_non_speech_segments.shape[0] > 0:
                speech_segments = torch.cat((speech_segments, short_non_speech_segments), 0)
                speech_segments = merge_overlap_segment(speech_segments)
    else:
        if min_duration_off > 0.0 and speech_segments.shape[0] > 0:
            non_speech_segments = get_gap_segments(speech_segments)
            short_non_speech_segments = remove_segments(
                non_speech_segments, filter_short_segments(non_speech_segments, min_duration_off)
            )
            if short_non_speech_segments.shape[0] > 0:
                speech_segments = torch.cat((speech_segments, short_non_speech_segments), 0)
                speech_segments = merge_overlap_segment(speech_segments)
        if min_duration_on > 0.0:
            speech_segments = filter_short_segments(speech_segments, min_duration_on)

    return speech_segments

def run_case(name, probs, per_args, frame_length=0.01):
    """Run NeMo binarization + filtering and return segments."""
    per_args_with_frame = {**per_args, 'frame_length_in_sec': frame_length}
    seq = torch.tensor(probs, dtype=torch.float32)

    segments = binarization(seq, per_args_with_frame)
    segments = filtering(segments, per_args_with_frame)

    if segments.shape == torch.Size([0]):
        return {"name": name, "segments": []}

    segments, _ = torch.sort(segments, 0)
    result = []
    for i in range(segments.shape[0]):
        result.append({
            "start": round(float(segments[i, 0]), 6),
            "end": round(float(segments[i, 1]), 6),
        })
    return {"name": name, "segments": result}


# DIHARD3 parameters
dihard3 = {
    "onset": 0.56,
    "offset": 1.0,
    "pad_onset": 0.063,
    "pad_offset": 0.002,
    "min_duration_on": 0.007,   # short non-speech deletion
    "min_duration_off": 0.151,  # short speech deletion
}

# CallHome parameters
callhome = {
    "onset": 0.641,
    "offset": 0.561,
    "pad_onset": 0.229,
    "pad_offset": 0.079,
    "min_duration_on": 0.511,
    "min_duration_off": 0.296,
}

# Frame duration: Sortformer outputs at 80ms per frame
FRAME_LEN = 0.08

cases = []

# Case 1: Single speaker, clean speech (DIHARD3)
probs1 = [0.0]*5 + [0.9]*15 + [0.0]*10
cases.append(run_case("single_speaker_dihard3", probs1, dihard3, FRAME_LEN))

# Case 2: Two speakers alternating (DIHARD3)
probs2_s0 = [0.0]*3 + [0.9]*10 + [0.0]*5 + [0.9]*8 + [0.0]*4
probs2_s1 = [0.0]*15 + [0.8]*10 + [0.0]*5
cases.append(run_case("speaker0_dihard3", probs2_s0, dihard3, FRAME_LEN))
cases.append(run_case("speaker1_dihard3", probs2_s1, dihard3, FRAME_LEN))

# Case 3: Short speech that should be filtered (DIHARD3)
# 1 frame = 0.08s, with padding = 0.08 + 0.063 + 0.002 = 0.145s < 0.151s
probs3 = [0.0]*5 + [0.9]*1 + [0.0]*10
cases.append(run_case("too_short_dihard3", probs3, dihard3, FRAME_LEN))

# Case 4: Short silence gap that should be filled (DIHARD3)
# min_duration_on=0.007 means gaps < 0.007s are filled
# At 80ms frames, even 1 frame gap (0.08s) > 0.007s, so gaps always survive
# But with offset=1.0, every non-1.0 frame triggers segment end...
# Let's see what NeMo actually produces
probs4 = [0.0]*3 + [0.9]*5 + [0.1]*1 + [0.9]*5 + [0.0]*6
cases.append(run_case("short_gap_dihard3", probs4, dihard3, FRAME_LEN))

# Case 5: Continuous speech with varying confidence (DIHARD3)
probs5 = [0.0]*2 + [0.7, 0.9, 0.8, 0.6, 0.9, 0.95, 0.85, 0.7] + [0.0]*5
cases.append(run_case("varying_confidence_dihard3", probs5, dihard3, FRAME_LEN))

# Case 6: offset=1.0 behavior — prob=0.9 is < 1.0, what happens?
probs6 = [0.0]*2 + [0.9]*10 + [0.0]*8
cases.append(run_case("offset1_behavior", probs6, dihard3, FRAME_LEN))

# Case 7: Same as case 6 but with CallHome params for comparison
cases.append(run_case("offset1_callhome", probs6, callhome, FRAME_LEN))

# Case 8: All silence
probs8 = [0.1]*20
cases.append(run_case("all_silence", probs8, dihard3, FRAME_LEN))

# Case 9: Overlap region (both speakers high)
probs9 = [0.0]*2 + [0.9]*8 + [0.0]*10
cases.append(run_case("overlap_spk", probs9, dihard3, FRAME_LEN))

# Case 10: Realistic long sequence with multiple segments
import random
random.seed(42)
probs10 = []
for i in range(200):
    if 10 <= i < 30 or 50 <= i < 80 or 120 <= i < 160:
        probs10.append(0.6 + random.random() * 0.35)
    else:
        probs10.append(random.random() * 0.2)
cases.append(run_case("realistic_long_dihard3", probs10, dihard3, FRAME_LEN))

output = {"frame_length": FRAME_LEN, "cases": cases}

# Also store the raw probs for Swift to use
raw_probs = {
    "single_speaker_dihard3": probs1,
    "speaker0_dihard3": probs2_s0,
    "speaker1_dihard3": probs2_s1,
    "too_short_dihard3": probs3,
    "short_gap_dihard3": probs4,
    "varying_confidence_dihard3": probs5,
    "offset1_behavior": probs6,
    "offset1_callhome": probs6,
    "all_silence": probs8,
    "overlap_spk": probs9,
    "realistic_long_dihard3": probs10,
}
output["raw_probs"] = raw_probs
output["dihard3_params"] = dihard3
output["callhome_params"] = callhome

with open("scripts/nemo_binarization_truth.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Generated {len(cases)} test cases\n")
for c in cases:
    segs = c["segments"]
    print(f"  {c['name']:35s}  {len(segs)} segments")
    for s in segs:
        print(f"    [{s['start']:.4f}, {s['end']:.4f}]")
