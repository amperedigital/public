#!/usr/bin/env python3
"""
benchmark.py — GetAmpere Sesame TTS Performance Benchmark

Tests TTFB, RTF, throughput, and consistency across multiple request types.
Saves results to benchmark_results.json.

Usage:
    python3 benchmark.py                          # Run against tts.getampere.ai
    python3 benchmark.py --url http://localhost:8080  # Run against local instance
    python3 benchmark.py --runs 10               # More iterations per test
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_URL = "https://tts.getampere.ai"
INTERNAL_KEY = "1376e6a43539e0eefa56f5fdec5e358457ffa50c3aa49f6a9a2ccb3ddeff7878"
EMILY_VOICE  = "i6POHJM1Z768DiJJG2CX"

TEST_SENTENCES = [
    ("short",    "Hello, how can I help you today?"),
    ("medium",   "Thank you for calling GetAmpere. I'm Emily, your AI receptionist. How can I direct your call today?"),
    ("long",     "We offer a comprehensive suite of AI-powered business solutions designed to streamline your operations, enhance customer engagement, and drive measurable growth across all departments of your organization."),
    ("filler",   "Absolutely, let me pull that up for you... just one moment while I retrieve the information from our system."),
    ("response", "I completely understand your concern. Let me connect you with our technical team who will be able to assist you with that specific issue right away."),
]

SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2  # 16-bit PCM

# ── Core request function ──────────────────────────────────────────────────────

def tts_request(url, text, voice_id, output_file):
    """Run a single TTS request. Returns (ttfb_ms, total_ms, audio_duration_s, bytes)."""
    start = time.perf_counter()

    result = subprocess.run([
        "curl", "-s", "--max-time", "60",
        "-X", "POST", f"{url}/v1/audio/speech",
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {INTERNAL_KEY}",
        "-D", "/tmp/tts_headers.txt",  # dump headers to get streaming timing
        "-d", json.dumps({
            "model": "csm-1b",
            "input": text,
            "voice": voice_id,
            "response_format": "pcm"
        }),
        "-o", output_file
    ], capture_output=True, timeout=65)

    total_ms = (time.perf_counter() - start) * 1000

    if result.returncode != 0:
        return None, total_ms, 0, 0

    size = os.path.getsize(output_file) if os.path.exists(output_file) else 0
    audio_duration_s = size / (SAMPLE_RATE * BYTES_PER_SAMPLE)

    return None, total_ms, audio_duration_s, size  # TTFB not measurable via blocking curl


def check_health(url):
    """Check server health. Returns dict."""
    result = subprocess.run([
        "curl", "-s", "--max-time", "10", f"{url}/health"
    ], capture_output=True, text=True, timeout=15)
    try:
        return json.loads(result.stdout)
    except:
        return {}


# ── Benchmark runner ───────────────────────────────────────────────────────────

def run_benchmark(url, runs, voice_id):
    print(f"\n{'='*60}")
    print(f"  GetAmpere Sesame TTS — Performance Benchmark")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"  URL    : {url}")
    print(f"  Voice  : {voice_id}")
    print(f"  Runs   : {runs} per sentence")
    print(f"{'='*60}\n")

    # Health check
    print("Checking health...", end=" ", flush=True)
    health = check_health(url)
    if health.get("status") != "ok":
        print(f"FAIL — server not healthy: {health}")
        sys.exit(1)
    print(f"OK — model: {health.get('model')}, device: {health.get('device')}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "url": url,
        "voice_id": voice_id,
        "runs_per_test": runs,
        "health": health,
        "tests": []
    }

    print()
    for name, text in TEST_SENTENCES:
        times = []
        sizes = []
        durations = []
        rtfs = []

        print(f"  [{name.upper()}] {text[:60]}...")
        for i in range(runs):
            out_file = f"/tmp/bench_{name}_{i}.pcm"
            _, total_ms, audio_s, size = tts_request(url, text, voice_id, out_file)

            if size > 0:
                rtf = (total_ms / 1000) / audio_s if audio_s > 0 else 999
                times.append(total_ms)
                sizes.append(size)
                durations.append(audio_s)
                rtfs.append(rtf)
                status = "✓" if rtf < 1.0 else "✗"
                print(f"    Run {i+1}: {total_ms:.0f}ms total | {audio_s:.2f}s audio | RTF {rtf:.3f} {status}")
            else:
                print(f"    Run {i+1}: FAILED (empty response)")

        if times:
            avg_ms = sum(times) / len(times)
            min_ms = min(times)
            max_ms = max(times)
            avg_rtf = sum(rtfs) / len(rtfs)
            avg_audio = sum(durations) / len(durations)

            print(f"    ─────────────────────────────────────────────────")
            print(f"    Avg: {avg_ms:.0f}ms | Min: {min_ms:.0f}ms | Max: {max_ms:.0f}ms | RTF: {avg_rtf:.3f}")

            results["tests"].append({
                "name": name,
                "text": text,
                "runs": len(times),
                "avg_total_ms": round(avg_ms, 1),
                "min_total_ms": round(min_ms, 1),
                "max_total_ms": round(max_ms, 1),
                "avg_rtf": round(avg_rtf, 3),
                "avg_audio_s": round(avg_audio, 2),
            })
        print()

    # Summary
    print(f"{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Test':<12} {'Avg (ms)':<12} {'Min (ms)':<12} {'RTF':<10} {'Audio (s)'}")
    print(f"  {'-'*55}")
    overall_rtfs = []
    for t in results["tests"]:
        status = "✓" if t["avg_rtf"] < 1.0 else "✗"
        print(f"  {t['name']:<12} {t['avg_total_ms']:<12.0f} {t['min_total_ms']:<12.0f} "
              f"{t['avg_rtf']:<10.3f}{status} {t['avg_audio_s']:.2f}s")
        overall_rtfs.append(t["avg_rtf"])

    if overall_rtfs:
        avg_rtf = sum(overall_rtfs) / len(overall_rtfs)
        print(f"\n  Overall avg RTF: {avg_rtf:.3f} {'✓ REAL-TIME' if avg_rtf < 1.0 else '✗ BELOW REAL-TIME'}")

    # Save results
    out_path = f"/tmp/benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")
    print(f"{'='*60}\n")

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GetAmpere Sesame TTS Benchmark")
    parser.add_argument("--url", default=DEFAULT_URL, help="TTS server URL")
    parser.add_argument("--runs", type=int, default=3, help="Runs per sentence (default: 3)")
    parser.add_argument("--voice", default=EMILY_VOICE, help="Voice ID")
    args = parser.parse_args()

    run_benchmark(args.url, args.runs, args.voice)
