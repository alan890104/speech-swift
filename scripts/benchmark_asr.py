#!/usr/bin/env python3
"""
ASR WER benchmark for speech-swift.

Downloads LibriSpeech test-clean, runs transcription via CLI, computes WER.

Usage:
    python scripts/benchmark_asr.py [--engine qwen3] [--model 0.6B] [--num-files 10]
    python scripts/benchmark_asr.py --download-only
    python scripts/benchmark_asr.py --score-only
    python scripts/benchmark_asr.py --compare
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"

BENCHMARK_DIR = Path("benchmarks/librispeech")
DATA_DIR = BENCHMARK_DIR / "test-clean"
HYP_DIR = BENCHMARK_DIR / "hyp"
RESULTS_FILE = BENCHMARK_DIR / "results.json"


# ---------------------------------------------------------------------------
# Text normalization & WER
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> dict:
    """Word Error Rate via edit distance with S/I/D breakdown."""
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()

    n = len(ref)
    m = len(hyp)

    # DP table
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    # Backtrace for S/I/D
    subs, ins, dels = 0, 0, 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            dels += 1
            i -= 1

    errors = subs + ins + dels
    wer = errors / max(n, 1) * 100

    return {
        "wer": round(wer, 2),
        "errors": errors,
        "substitutions": subs,
        "insertions": ins,
        "deletions": dels,
        "ref_words": n,
        "hyp_words": m,
    }


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_librispeech():
    """Download and extract LibriSpeech test-clean (~350 MB)."""
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    tar_path = BENCHMARK_DIR / "test-clean.tar.gz"
    if DATA_DIR.exists() and any(DATA_DIR.rglob("*.flac")):
        print(f"LibriSpeech test-clean already extracted at {DATA_DIR}")
        return

    if not tar_path.exists():
        print(f"Downloading LibriSpeech test-clean (~350 MB)...")
        print(f"  From: {LIBRISPEECH_URL}")
        print(f"  To:   {tar_path}")
        try:
            subprocess.run(
                ["curl", "-L", "-o", str(tar_path), "--progress-bar",
                 LIBRISPEECH_URL],
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  Download failed: {e}")
            print(f"  Download manually: curl -L -o {tar_path} '{LIBRISPEECH_URL}'")
            sys.exit(1)

    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tf:
        # LibriSpeech tar extracts to LibriSpeech/test-clean/...
        tf.extractall(BENCHMARK_DIR)

    # Move LibriSpeech/test-clean → benchmarks/librispeech/test-clean
    extracted = BENCHMARK_DIR / "LibriSpeech" / "test-clean"
    if extracted.exists() and not DATA_DIR.exists():
        extracted.rename(DATA_DIR)
    # Clean up empty LibriSpeech dir
    ls_dir = BENCHMARK_DIR / "LibriSpeech"
    if ls_dir.exists() and not any(ls_dir.iterdir()):
        ls_dir.rmdir()

    flac_count = len(list(DATA_DIR.rglob("*.flac")))
    print(f"  Extracted {flac_count} FLAC files")


def load_transcripts() -> list:
    """Parse LibriSpeech transcript files. Returns [(utterance_id, flac_path, text)]."""
    utterances = []
    for trans_file in sorted(DATA_DIR.rglob("*.trans.txt")):
        chapter_dir = trans_file.parent
        for line in trans_file.read_text().strip().split("\n"):
            parts = line.strip().split(" ", 1)
            if len(parts) != 2:
                continue
            utt_id, text = parts
            flac_path = chapter_dir / f"{utt_id}.flac"
            if flac_path.exists():
                utterances.append((utt_id, str(flac_path), text))
    return sorted(utterances, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_file(cli_path: str, audio_path: str, engine: str,
                    model: str, timeout: int = 120) -> dict:
    """Run CLI transcription on a single file. Returns parsed result."""
    cmd = [cli_path, "transcribe", audio_path, "--engine", engine]
    if engine in ("qwen3", "qwen3-coreml", "qwen3-coreml-full"):
        cmd.extend(["--model", model])

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout
    )

    if result.returncode != 0:
        return {"error": result.stderr.strip()[:200]}

    text = ""
    rtf = 0.0
    inference_time = 0.0

    for line in result.stdout.split("\n"):
        if line.startswith("Result: "):
            text = line[len("Result: "):]
        elif "Time:" in line and "RTF:" in line:
            m = re.search(r"Time:\s*([\d.]+)s.*RTF:\s*([\d.]+)", line)
            if m:
                inference_time = float(m.group(1))
                rtf = float(m.group(2))

    return {"text": text, "rtf": rtf, "inference_time": inference_time}


def run_transcriptions(cli_path: str, utterances: list, engine: str,
                       model: str, timeout: int = 120) -> list:
    """Transcribe all utterances and return per-file results."""
    HYP_DIR.mkdir(parents=True, exist_ok=True)
    hyp_subdir = HYP_DIR / f"{engine}_{model}"
    hyp_subdir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(utterances)
    failures = 0

    for idx, (utt_id, flac_path, ref_text) in enumerate(utterances):
        pct = (idx + 1) / total * 100
        print(f"\r  [{idx+1}/{total}] ({pct:.0f}%) {utt_id}...",
              end="", flush=True)

        try:
            out = transcribe_file(cli_path, flac_path, engine, model, timeout)
        except subprocess.TimeoutExpired:
            out = {"error": "timeout"}
        except Exception as e:
            out = {"error": str(e)}

        if "error" in out:
            failures += 1
            results.append({
                "utterance_id": utt_id,
                "error": out["error"],
            })
            continue

        # Save hypothesis
        (hyp_subdir / f"{utt_id}.txt").write_text(out["text"])

        # Score
        wer_result = compute_wer(ref_text, out["text"])

        results.append({
            "utterance_id": utt_id,
            "reference": normalize_text(ref_text),
            "hypothesis": normalize_text(out["text"]),
            "wer": wer_result["wer"],
            "substitutions": wer_result["substitutions"],
            "insertions": wer_result["insertions"],
            "deletions": wer_result["deletions"],
            "ref_words": wer_result["ref_words"],
            "rtf": out["rtf"],
            "inference_time": out["inference_time"],
        })

    print()  # newline after progress
    if failures:
        print(f"  {failures} utterances failed")

    return results


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_existing(engine: str, model: str) -> list:
    """Re-score existing hypothesis files."""
    hyp_subdir = HYP_DIR / f"{engine}_{model}"
    if not hyp_subdir.exists():
        print(f"No hypothesis directory: {hyp_subdir}")
        return []

    utterances = load_transcripts()
    ref_map = {u[0]: u[2] for u in utterances}

    results = []
    for hyp_file in sorted(hyp_subdir.glob("*.txt")):
        utt_id = hyp_file.stem
        if utt_id not in ref_map:
            continue
        hyp_text = hyp_file.read_text().strip()
        ref_text = ref_map[utt_id]
        wer_result = compute_wer(ref_text, hyp_text)
        results.append({
            "utterance_id": utt_id,
            "reference": normalize_text(ref_text),
            "hypothesis": normalize_text(hyp_text),
            "wer": wer_result["wer"],
            "substitutions": wer_result["substitutions"],
            "insertions": wer_result["insertions"],
            "deletions": wer_result["deletions"],
            "ref_words": wer_result["ref_words"],
        })

    return results


def aggregate_results(per_file: list, engine: str, model: str) -> dict:
    """Compute aggregate WER from per-file results."""
    scored = [r for r in per_file if "error" not in r]
    failed = [r for r in per_file if "error" in r]

    total_ref = sum(r["ref_words"] for r in scored)
    total_sub = sum(r.get("substitutions", 0) for r in scored)
    total_ins = sum(r.get("insertions", 0) for r in scored)
    total_del = sum(r.get("deletions", 0) for r in scored)
    total_errors = total_sub + total_ins + total_del
    total_time = sum(r.get("inference_time", 0) for r in scored)

    agg_wer = total_errors / max(total_ref, 1) * 100

    result = {
        "engine": engine,
        "model": model,
        "dataset": "librispeech-test-clean",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_utterances": len(scored),
        "num_failures": len(failed),
        "aggregate_wer": round(agg_wer, 2),
        "total_ref_words": total_ref,
        "total_substitutions": total_sub,
        "total_insertions": total_ins,
        "total_deletions": total_del,
        "total_inference_time_s": round(total_time, 2),
        "per_file": per_file,
    }

    # RTF if available
    rtfs = [r["rtf"] for r in scored if r.get("rtf", 0) > 0]
    if rtfs:
        result["mean_rtf"] = round(sum(rtfs) / len(rtfs), 4)

    return result


def print_summary(results: dict):
    """Print summary table."""
    print(f"\n{'='*60}")
    print(f"ASR Benchmark: {results['dataset']}")
    print(f"Engine: {results['engine']}, Model: {results['model']}")
    print(f"{'='*60}")
    print(f"  Utterances:     {results['num_utterances']}"
          f" ({results['num_failures']} failed)")
    print(f"  Aggregate WER:  {results['aggregate_wer']:.2f}%")
    print(f"  Total words:    {results['total_ref_words']}")
    print(f"  Substitutions:  {results['total_substitutions']}")
    print(f"  Insertions:     {results['total_insertions']}")
    print(f"  Deletions:      {results['total_deletions']}")
    if "mean_rtf" in results:
        print(f"  Mean RTF:       {results['mean_rtf']:.4f}")
    print(f"{'='*60}")

    # Show worst utterances
    scored = [r for r in results["per_file"] if "error" not in r and r["wer"] > 0]
    if scored:
        worst = sorted(scored, key=lambda x: x["wer"], reverse=True)[:10]
        print(f"\nWorst 10 utterances:")
        for r in worst:
            print(f"  {r['utterance_id']}: WER={r['wer']:.1f}%")
            print(f"    ref: {r['reference']}")
            print(f"    hyp: {r['hypothesis']}")


# ---------------------------------------------------------------------------
# Multi-engine comparison
# ---------------------------------------------------------------------------

ENGINE_CONFIGS = [
    ("qwen3", "0.6B"),
    ("qwen3", "0.6B-8bit"),
    ("qwen3", "1.7B"),
    ("qwen3", "1.7B-4bit"),
    ("parakeet", "default"),
]


def run_comparison(cli_path: str, utterances: list, timeout: int = 120):
    """Run all engine/model combinations and print comparison table."""
    all_results = []

    for engine, model in ENGINE_CONFIGS:
        print(f"\n--- {engine} / {model} ---")
        per_file = run_transcriptions(
            cli_path, utterances, engine, model, timeout)
        agg = aggregate_results(per_file, engine, model)
        all_results.append(agg)
        print_summary(agg)

        # Save per-engine results
        out_file = BENCHMARK_DIR / f"results_{engine}_{model}.json"
        with open(out_file, "w") as f:
            json.dump(agg, f, indent=2)

    # Comparison table
    print(f"\n{'='*60}")
    print(f"{'Engine':<20} {'Model':<12} {'WER%':>8} {'RTF':>8}")
    print(f"{'-'*20} {'-'*12} {'-'*8} {'-'*8}")
    for r in all_results:
        rtf_str = f"{r['mean_rtf']:.4f}" if "mean_rtf" in r else "N/A"
        print(f"{r['engine']:<20} {r['model']:<12} {r['aggregate_wer']:>7.2f}% {rtf_str:>8}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ASR WER benchmark (LibriSpeech test-clean)")
    parser.add_argument("--cli-path", default=".build/release/audio",
                        help="Path to audio CLI binary")
    parser.add_argument("--engine", default="qwen3",
                        help="ASR engine: qwen3, parakeet, qwen3-coreml, "
                             "qwen3-coreml-full")
    parser.add_argument("--model", default="0.6B",
                        help="Model variant: 0.6B, 0.6B-8bit, 1.7B, 1.7B-4bit")
    parser.add_argument("--num-files", type=int, default=0,
                        help="Limit number of utterances (0 = all)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Per-file timeout in seconds")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download and extract test data")
    parser.add_argument("--score-only", action="store_true",
                        help="Re-score existing hypothesis transcriptions")
    parser.add_argument("--compare", action="store_true",
                        help="Run all engine/model combinations")
    args = parser.parse_args()

    # Download
    if not args.score_only:
        download_librispeech()

    if args.download_only:
        print("\nDownload complete.")
        return

    # Load transcripts
    utterances = load_transcripts()
    if not utterances:
        print("No transcripts found. Run with --download-only first.")
        sys.exit(1)

    if args.num_files > 0:
        utterances = utterances[:args.num_files]
    print(f"Loaded {len(utterances)} utterances")

    # Comparison mode
    if args.compare:
        if not Path(args.cli_path).exists():
            print(f"CLI not found: {args.cli_path}. Build with: make build")
            sys.exit(1)
        run_comparison(args.cli_path, utterances, args.timeout)
        return

    # Single engine mode
    if args.score_only:
        per_file = score_existing(args.engine, args.model)
    else:
        if not Path(args.cli_path).exists():
            print(f"CLI not found: {args.cli_path}. Build with: make build")
            sys.exit(1)
        print(f"\nTranscribing with {args.engine}/{args.model}...")
        per_file = run_transcriptions(
            args.cli_path, utterances, args.engine, args.model, args.timeout)

    if not per_file:
        print("No results to score.")
        return

    results = aggregate_results(per_file, args.engine, args.model)
    print_summary(results)

    # Save
    out_file = BENCHMARK_DIR / f"results_{args.engine}_{args.model}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
