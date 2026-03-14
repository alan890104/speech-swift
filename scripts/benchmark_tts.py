#!/usr/bin/env python3
"""
TTS round-trip WER benchmark for speech-swift.

Synthesizes text, transcribes the audio back, computes WER vs original.
Measures TTS intelligibility end-to-end.

Usage:
    python scripts/benchmark_tts.py [--tts-engine qwen3] [--num-sentences 10]
    python scripts/benchmark_tts.py --compare
    python scripts/benchmark_tts.py --input-file sentences.txt
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

BENCHMARK_DIR = Path("benchmarks/tts")

# Built-in test corpus: diverse phonemes, lengths, punctuation
TEST_SENTENCES = [
    # Short
    "Hello world.",
    "Good morning everyone.",
    "Thank you very much.",
    "What time is it?",
    "Nice to meet you.",
    # Medium
    "The quick brown fox jumps over the lazy dog.",
    "Can you guarantee that the replacement part will be shipped tomorrow?",
    "The weather is beautiful today, perfect for a walk in the park.",
    "Please make sure to send the report before the end of the day.",
    "I would like to schedule a meeting for next Wednesday afternoon.",
    # Long
    "Scientists have discovered a new species of deep sea fish that can survive "
    "at extreme pressures found at the bottom of the ocean.",
    "The development team has been working around the clock to deliver the new "
    "software update, which includes several critical bug fixes and performance "
    "improvements.",
    "After careful consideration of all the available evidence, the committee "
    "decided to postpone the decision until the next quarterly review.",
    # Numbers and special content
    "The temperature today is twenty three degrees celsius.",
    "Our company was founded in nineteen ninety nine.",
    "The flight departs at three forty five in the afternoon.",
    # Questions and commands
    "Could you please explain how this algorithm works?",
    "Turn left at the next intersection and continue for two miles.",
    "Have you ever visited the national museum of natural history?",
    "Remember to water the plants every other day during the summer.",
    # Technical
    "Machine learning models require large amounts of training data.",
    "The server response time should be under two hundred milliseconds.",
    "Cloud computing enables on demand access to shared resources.",
    "Natural language processing is a subfield of artificial intelligence.",
    # Conversational
    "I think we should take a different approach to this problem.",
    "That sounds like a great idea, let me think about it.",
    "Sorry, I did not catch what you said, could you repeat that?",
    "The restaurant around the corner serves excellent Italian food.",
    "We need to finish this project before the deadline next Friday.",
    "Let me know if you have any questions about the presentation.",
]


# ---------------------------------------------------------------------------
# Text normalization & WER (same as benchmark_asr.py)
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
# TTS + ASR pipeline
# ---------------------------------------------------------------------------

def synthesize(cli_path: str, text: str, output_path: str,
               engine: str, model: str, timeout: int = 180) -> dict:
    """Synthesize text to audio file. Returns timing info."""
    if engine == "kokoro":
        cmd = [cli_path, "kokoro", text, "--output", output_path]
    else:
        cmd = [cli_path, "speak", text, "--output", output_path,
               "--engine", engine]
        if engine == "qwen3" and model != "default":
            cmd.extend(["--model", model])

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout
    )

    if result.returncode != 0:
        return {"error": result.stderr.strip()[:200]}

    tts_rtf = 0.0
    tts_time = 0.0
    audio_duration = 0.0

    for line in result.stdout.split("\n"):
        m = re.search(
            r"Duration:\s*([\d.]+)s.*Time:\s*([\d.]+)s.*RTF:\s*([\d.]+)",
            line)
        if m:
            audio_duration = float(m.group(1))
            tts_time = float(m.group(2))
            tts_rtf = float(m.group(3))

    return {
        "tts_rtf": tts_rtf,
        "tts_time": tts_time,
        "audio_duration": audio_duration,
    }


def transcribe(cli_path: str, audio_path: str, asr_engine: str,
               asr_model: str, timeout: int = 120) -> dict:
    """Transcribe audio file. Returns text and timing."""
    cmd = [cli_path, "transcribe", audio_path, "--engine", asr_engine]
    if asr_engine in ("qwen3", "qwen3-coreml", "qwen3-coreml-full"):
        cmd.extend(["--model", asr_model])

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout
    )

    if result.returncode != 0:
        return {"error": result.stderr.strip()[:200]}

    text = ""
    asr_rtf = 0.0

    for line in result.stdout.split("\n"):
        if line.startswith("Result: "):
            text = line[len("Result: "):]
        elif "Time:" in line and "RTF:" in line:
            m = re.search(r"RTF:\s*([\d.]+)", line)
            if m:
                asr_rtf = float(m.group(1))

    return {"text": text, "asr_rtf": asr_rtf}


def run_benchmark(cli_path: str, sentences: list,
                  tts_engine: str, tts_model: str,
                  asr_engine: str, asr_model: str,
                  timeout: int = 180) -> list:
    """Run TTS round-trip benchmark on all sentences."""
    results = []
    total = len(sentences)

    with tempfile.TemporaryDirectory(prefix="tts_bench_") as tmpdir:
        for idx, text in enumerate(sentences):
            pct = (idx + 1) / total * 100
            short = text[:50] + "..." if len(text) > 50 else text
            print(f"\r  [{idx+1}/{total}] ({pct:.0f}%) {short}",
                  end="", flush=True)

            wav_path = os.path.join(tmpdir, f"tts_{idx:04d}.wav")

            # Synthesize
            try:
                tts_out = synthesize(
                    cli_path, text, wav_path, tts_engine, tts_model, timeout)
            except subprocess.TimeoutExpired:
                tts_out = {"error": "tts_timeout"}
            except Exception as e:
                tts_out = {"error": str(e)}

            if "error" in tts_out:
                results.append({
                    "index": idx,
                    "input_text": text,
                    "error": f"TTS: {tts_out['error']}",
                })
                continue

            if not os.path.exists(wav_path):
                results.append({
                    "index": idx,
                    "input_text": text,
                    "error": "TTS produced no output file",
                })
                continue

            # Transcribe
            try:
                asr_out = transcribe(
                    cli_path, wav_path, asr_engine, asr_model, timeout)
            except subprocess.TimeoutExpired:
                asr_out = {"error": "asr_timeout"}
            except Exception as e:
                asr_out = {"error": str(e)}

            if "error" in asr_out:
                results.append({
                    "index": idx,
                    "input_text": text,
                    "error": f"ASR: {asr_out['error']}",
                })
                continue

            # Score
            wer_result = compute_wer(text, asr_out["text"])

            results.append({
                "index": idx,
                "input_text": normalize_text(text),
                "transcription": normalize_text(asr_out["text"]),
                "wer": wer_result["wer"],
                "substitutions": wer_result["substitutions"],
                "insertions": wer_result["insertions"],
                "deletions": wer_result["deletions"],
                "ref_words": wer_result["ref_words"],
                "tts_rtf": tts_out["tts_rtf"],
                "tts_time": tts_out["tts_time"],
                "audio_duration": tts_out["audio_duration"],
                "asr_rtf": asr_out["asr_rtf"],
            })

    print()  # newline after progress
    return results


def aggregate_results(per_sentence: list, tts_engine: str, tts_model: str,
                      asr_engine: str, asr_model: str) -> dict:
    """Compute aggregate metrics."""
    scored = [r for r in per_sentence if "error" not in r]
    failed = [r for r in per_sentence if "error" in r]

    total_ref = sum(r["ref_words"] for r in scored)
    total_sub = sum(r.get("substitutions", 0) for r in scored)
    total_ins = sum(r.get("insertions", 0) for r in scored)
    total_del = sum(r.get("deletions", 0) for r in scored)
    total_errors = total_sub + total_ins + total_del

    agg_wer = total_errors / max(total_ref, 1) * 100

    result = {
        "tts_engine": tts_engine,
        "tts_model": tts_model,
        "asr_engine": asr_engine,
        "asr_model": asr_model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_sentences": len(scored),
        "num_failures": len(failed),
        "aggregate_wer": round(agg_wer, 2),
        "total_ref_words": total_ref,
        "total_substitutions": total_sub,
        "total_insertions": total_ins,
        "total_deletions": total_del,
        "per_sentence": per_sentence,
    }

    tts_rtfs = [r["tts_rtf"] for r in scored if r.get("tts_rtf", 0) > 0]
    if tts_rtfs:
        result["mean_tts_rtf"] = round(sum(tts_rtfs) / len(tts_rtfs), 4)

    return result


def print_summary(results: dict):
    """Print summary."""
    print(f"\n{'='*60}")
    print(f"TTS Round-Trip Benchmark")
    print(f"TTS: {results['tts_engine']}/{results['tts_model']}  "
          f"ASR: {results['asr_engine']}/{results['asr_model']}")
    print(f"{'='*60}")
    print(f"  Sentences:      {results['num_sentences']}"
          f" ({results['num_failures']} failed)")
    print(f"  Round-trip WER: {results['aggregate_wer']:.2f}%")
    print(f"  Total words:    {results['total_ref_words']}")
    print(f"  Substitutions:  {results['total_substitutions']}")
    print(f"  Insertions:     {results['total_insertions']}")
    print(f"  Deletions:      {results['total_deletions']}")
    if "mean_tts_rtf" in results:
        print(f"  Mean TTS RTF:   {results['mean_tts_rtf']:.4f}")
    print(f"{'='*60}")

    # Show errors
    scored = [r for r in results["per_sentence"] if "error" not in r]
    errors = [r for r in scored if r["wer"] > 0]
    if errors:
        print(f"\nMismatched sentences ({len(errors)}):")
        for r in sorted(errors, key=lambda x: x["wer"], reverse=True):
            print(f"  [{r['index']}] WER={r['wer']:.1f}%")
            print(f"    in:  {r['input_text']}")
            print(f"    out: {r['transcription']}")


# ---------------------------------------------------------------------------
# Multi-engine comparison
# ---------------------------------------------------------------------------

TTS_CONFIGS = [
    ("qwen3", "base"),
    ("qwen3", "base-8bit"),
    ("cosyvoice", "default"),
    ("kokoro", "default"),
]


def run_comparison(cli_path: str, sentences: list,
                   asr_engine: str, asr_model: str, timeout: int = 180):
    """Run all TTS engines and print comparison."""
    all_results = []

    for tts_engine, tts_model in TTS_CONFIGS:
        print(f"\n--- TTS: {tts_engine}/{tts_model} ---")
        per_sentence = run_benchmark(
            cli_path, sentences, tts_engine, tts_model,
            asr_engine, asr_model, timeout)
        agg = aggregate_results(
            per_sentence, tts_engine, tts_model, asr_engine, asr_model)
        all_results.append(agg)
        print_summary(agg)

        out_file = BENCHMARK_DIR / f"results_{tts_engine}_{tts_model}.json"
        with open(out_file, "w") as f:
            json.dump(agg, f, indent=2)

    # Comparison table
    print(f"\n{'='*60}")
    print(f"{'TTS Engine':<16} {'Model':<12} {'WER%':>8} {'TTS RTF':>10}")
    print(f"{'-'*16} {'-'*12} {'-'*8} {'-'*10}")
    for r in all_results:
        rtf_str = f"{r['mean_tts_rtf']:.4f}" if "mean_tts_rtf" in r else "N/A"
        print(f"{r['tts_engine']:<16} {r['tts_model']:<12} "
              f"{r['aggregate_wer']:>7.2f}% {rtf_str:>10}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TTS round-trip WER benchmark")
    parser.add_argument("--cli-path", default=".build/release/audio",
                        help="Path to audio CLI binary")
    parser.add_argument("--tts-engine", default="qwen3",
                        help="TTS engine: qwen3, cosyvoice, kokoro")
    parser.add_argument("--tts-model", default="base",
                        help="TTS model variant")
    parser.add_argument("--asr-engine", default="qwen3",
                        help="ASR engine for transcription")
    parser.add_argument("--asr-model", default="0.6B",
                        help="ASR model for transcription")
    parser.add_argument("--num-sentences", type=int, default=0,
                        help="Limit number of sentences (0 = all)")
    parser.add_argument("--input-file", type=str, default=None,
                        help="File with one sentence per line")
    parser.add_argument("--timeout", type=int, default=180,
                        help="Per-sentence timeout in seconds")
    parser.add_argument("--compare", action="store_true",
                        help="Run all TTS engines")
    args = parser.parse_args()

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    if not Path(args.cli_path).exists():
        print(f"CLI not found: {args.cli_path}. Build with: make build")
        sys.exit(1)

    # Load sentences
    if args.input_file:
        sentences = [
            l.strip() for l in Path(args.input_file).read_text().split("\n")
            if l.strip()
        ]
    else:
        sentences = TEST_SENTENCES

    if args.num_sentences > 0:
        sentences = sentences[:args.num_sentences]
    print(f"Loaded {len(sentences)} test sentences")

    # Comparison mode
    if args.compare:
        run_comparison(
            args.cli_path, sentences,
            args.asr_engine, args.asr_model, args.timeout)
        return

    # Single engine
    print(f"\nTTS: {args.tts_engine}/{args.tts_model}, "
          f"ASR: {args.asr_engine}/{args.asr_model}")
    per_sentence = run_benchmark(
        args.cli_path, sentences,
        args.tts_engine, args.tts_model,
        args.asr_engine, args.asr_model,
        args.timeout)

    results = aggregate_results(
        per_sentence, args.tts_engine, args.tts_model,
        args.asr_engine, args.asr_model)
    print_summary(results)

    out_file = BENCHMARK_DIR / f"results_{args.tts_engine}_{args.tts_model}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
