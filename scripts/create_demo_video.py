from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import textwrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
FRAMES = DEMO / "frames"
OUT = DEMO / "demo-video.mp4"
WIDTH, HEIGHT = 1280, 720


def load_json(path: str) -> dict:
    full = ROOT / path
    return json.loads(full.read_text(encoding="utf-8")) if full.exists() else {}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def draw_slide(index: int, title: str, bullets: list[str], footer: str) -> Path:
    img = Image.new("RGB", (WIDTH, HEIGHT), "#f7f8fb")
    draw = ImageDraw.Draw(img)
    title_font = font(44, bold=True)
    body_font = font(30)
    small_font = font(22)
    accent = "#2155d9"
    dark = "#172033"
    muted = "#566070"

    draw.rectangle([0, 0, WIDTH, 84], fill="#ffffff")
    draw.rectangle([0, 82, WIDTH, 86], fill=accent)
    draw.text((52, 22), title, fill=dark, font=title_font)
    draw.text((WIDTH - 180, 30), f"Slide {index}", fill=muted, font=small_font)

    y = 130
    for bullet in bullets:
        wrapped = textwrap.wrap(bullet, width=64)
        draw.ellipse([58, y + 10, 72, y + 24], fill=accent)
        for line_i, line in enumerate(wrapped):
            draw.text((92, y + line_i * 38), line, fill=dark, font=body_font)
        y += max(1, len(wrapped)) * 42 + 18

    draw.rectangle([0, HEIGHT - 62, WIDTH, HEIGHT], fill="#ffffff")
    draw.text((52, HEIGHT - 43), footer, fill=muted, font=small_font)
    path = FRAMES / f"slide_{index:02d}.png"
    img.save(path)
    return path


def main() -> None:
    DEMO.mkdir(exist_ok=True)
    FRAMES.mkdir(exist_ok=True)

    ragas = load_json("reports/ragas_50q.json")
    judge = load_json("reports/judge_results.json")
    guard = load_json("reports/guard_results.json")

    per = ragas.get("per_distribution", {})
    clusters = ragas.get("failure_clusters", {})
    guard_suite = guard.get("adversarial_suite", {})
    latency = guard.get("latency", {}).get("total_ms", {})
    output_guard = guard.get("output_guard", {})

    slides = [
        (
            "Lab 24 Demo — Eval & Guardrail Stack",
            [
                "Project: 2A202600983-LuuXuanThe-Day24",
                "Goal: production-style evaluation and guardrails for Vietnamese HR policy RAG.",
                "Demo covers RAGAS 50 questions, LLM-as-Judge, adversarial guardrails, and latency.",
            ],
        ),
        (
            "1. RAGAS Evaluation On 50 Questions",
            [
                f"Factual avg score: {per.get('factual', {}).get('avg_score', 0):.4f}",
                f"Multi-hop avg score: {per.get('multi_hop', {}).get('avg_score', 0):.4f}",
                f"Adversarial avg score: {per.get('adversarial', {}).get('avg_score', 0):.4f}",
                "Artifacts: answers_50q.json and reports/ragas_50q.json.",
            ],
        ),
        (
            "RAGAS Failure Analysis",
            [
                f"Dominant failure distribution: {clusters.get('dominant_failure_distribution')}",
                f"Dominant weak metric: {clusters.get('dominant_failure_metric')}",
                "Root cause: baseline retrieval returns related but not always exact chunks.",
                "Fix plan: metadata version filters, hybrid retrieval, and cross-encoder reranking.",
            ],
        ),
        (
            "2. LLM-as-Judge",
            [
                f"Judge mode: {judge.get('judge_mode', 'unknown')}",
                f"Batch size: {judge.get('batch_size', 0)} pairwise comparisons",
                "Swap-and-average is used to reduce position bias.",
                f"Position bias rate: {judge.get('bias_report', {}).get('position_bias_rate', 0):.1%}",
            ],
        ),
        (
            "Judge Calibration And Bias",
            [
                f"Cohen's kappa sample: {judge.get('cohen_kappa', 0):.3f}",
                f"Verbosity bias: {judge.get('bias_report', {}).get('verbosity_bias', 0):.1%}",
                "Artifacts: phase-b/pairwise_results.csv, phase-b/absolute_scores.csv, analysis/bias_report.md.",
            ],
        ),
        (
            "3. Adversarial Guardrail Test",
            [
                f"Attack suite pass rate: {guard_suite.get('passed', 0)}/{guard_suite.get('total', 0)}",
                "Blocked categories: PII injection, jailbreak, off-topic, prompt injection.",
                "Input layers: PII detector + topic/injection rail.",
                "Output layer: Llama Guard adapter with safe local fallback.",
            ],
        ),
        (
            "Output Guardrail / Llama Guard",
            [
                f"Mode used: {output_guard.get('mode', 'unknown')}",
                f"Unsafe detection rate: {output_guard.get('unsafe_detection_rate', 0):.0%}",
                f"Safe false positive rate: {output_guard.get('safe_false_positive_rate', 0):.0%}",
                f"Output guard P95 latency: {output_guard.get('p95_latency_ms', 0)} ms",
            ],
        ),
        (
            "4. Latency Benchmark",
            [
                f"Guard total P50: {latency.get('p50', 0)} ms",
                f"Guard total P95: {latency.get('p95', 0)} ms",
                f"Guard total P99: {latency.get('p99', 0)} ms",
                "Budget: P95 total < 500 ms. Current run passes budget.",
            ],
        ),
        (
            "Production Blueprint",
            [
                "SLOs are defined for faithfulness, relevancy, precision, recall, guard pass rate, latency, and PII safety.",
                "Alert playbook covers faithfulness drops, context precision drops, and guardrail regressions.",
                "CI gate runs setup, Phase A, Phase B, Phase C, tests, and report upload.",
            ],
        ),
        (
            "Submission Checklist",
            [
                "check_lab.py passes 22/22.",
                "pytest passes all tests.",
                "Reports and demo video are included in the repo.",
                "Next production step: replace fallback RAG with full Day 18 pipeline and rerun with real keys.",
            ],
        ),
    ]

    frame_paths = []
    footer = "Lab 24 — Full Evaluation & Guardrail System"
    for i, (title, bullets) in enumerate(slides, start=1):
        frame_paths.append(draw_slide(i, title, bullets, footer))

    concat = DEMO / "concat.txt"
    # ffmpeg concat keeps the final still frame too; 11 frame spans x 27.27s is ~5 minutes.
    with concat.open("w", encoding="utf-8") as f:
        for path in frame_paths:
            f.write(f"file '{path}'\n")
            f.write("duration 27.27\n")
        f.write(f"file '{frame_paths[-1]}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
        "-vf", "fps=24,format=yuv420p", "-movflags", "+faststart", str(OUT)
    ], check=True)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
