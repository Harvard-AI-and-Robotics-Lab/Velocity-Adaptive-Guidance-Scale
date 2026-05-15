"""
Assess the difficulty level of prompts using the OpenAI API.

Reads all `[prompt_id].txt` files from an input folder, asks the model to rate
each prompt's difficulty on a 0-5 scale, and writes the results to a CSV.

Difficulty scale:
    0 = Very easy
    1 = Easy
    2 = Moderate
    3 = Difficult
    4 = Very difficult
    5 = Expert-level

Usage:
    export OPENAI_API_KEY="sk-..."
    python assess_prompt_difficulty.py \
        --input-dir ./prompts \
        --output-csv ./difficulty.csv \
        --model gpt-5.4

Requirements:
    pip install openai>=1.0.0

pip install openai
export OPENAI_API_KEY="sk-..."
python assess_prompt_difficulty.py --input-dir /path/to/datasets/coco17/validation_bestcaption --output-csv ./data4paper/prompt_difficulty_coco17.csv --model gpt-5.4

"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: The 'openai' package is required. Install with: pip install openai", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DIFFICULTY_LABELS = {
    0: "Very easy",
    1: "Easy",
    2: "Moderate",
    3: "Difficult",
    4: "Very difficult",
    5: "Expert-level",
}

SYSTEM_PROMPT = """You are an expert evaluator of text-to-image prompts.

Given one user prompt, rate how difficult it would be for a capable general-purpose image-generation model to faithfully generate the requested image.

Judge difficulty based on:
- Visual complexity: objects, people, actions, background, and scene density
- Composition: spatial relationships, perspective, occlusion, symmetry, or multi-panel layout
- Precision: required colors, textures, lighting, style, pose, camera angle, or realism
- Knowledge: specialized, cultural, historical, fictional, scientific, or technical concepts
- Constraints: number of details that must be simultaneously correct
- Text/diagram needs: readable words, labels, numbers, UI, charts, maps, or formulas
- Ambiguity: vague, conflicting, or underspecified instructions

Use this scale:
0 = Very easy: one simple common subject, minimal constraints
1 = Easy: common subject or scene with a few simple attributes
2 = Moderate: several elements or constraints, recognizable style, ordinary composition
3 = Difficult: complex scene, interacting subjects, precise composition, or specific style
4 = Very difficult: many constraints, rare concepts, technical accuracy, readable text, or unusual perspective
5 = Expert-level: extremely constrained, highly precise, contradictory, or requires domain-level visual accuracy

Important:
Judge image-generation difficulty, not prompt-reading difficulty. A vague prompt can be easy if many outputs would satisfy it. Increase difficulty for exact text, diagrams, hands, faces, anatomy, maps, charts, or scientific accuracy.

Respond ONLY as JSON:
{"difficulty": <integer 0-5>, "rationale": "<one short sentence>"}
"""

USER_TEMPLATE = "Assess the difficulty of the following prompt:\n\n---\n{prompt}\n---"

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class AssessmentResult:
    prompt_id: str
    prompt: str
    difficulty: Optional[int]
    rationale: str
    error: str = ""


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def parse_difficulty(raw_text: str) -> tuple[Optional[int], str]:
    """Extract difficulty integer and rationale from the model's response."""
    text = raw_text.strip()

    # Strip markdown code fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    # Try strict JSON first.
    try:
        obj = json.loads(text)
        difficulty = int(obj["difficulty"])
        rationale = str(obj.get("rationale", "")).strip()
        if 0 <= difficulty <= 5:
            return difficulty, rationale
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass

    # Fallback: find the first integer 0-5 anywhere in the response.
    match = re.search(r"\b([0-5])\b", text)
    if match:
        return int(match.group(1)), text[:200]

    return None, f"Could not parse: {text[:200]}"


def assess_one_prompt(
    client: OpenAI,
    model: str,
    prompt_id: str,
    prompt_text: str,
) -> AssessmentResult:
    """Send a single prompt to the API and return the assessment."""
    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(prompt=prompt_text)},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            difficulty, rationale = parse_difficulty(content)
            if difficulty is None:
                last_error = f"Unparseable response: {rationale}"
                continue
            return AssessmentResult(
                prompt_id=prompt_id,
                prompt=prompt_text,
                difficulty=difficulty,
                rationale=rationale,
            )
        except Exception as exc:  # noqa: BLE001 - we want any API failure to retry
            last_error = f"{type(exc).__name__}: {exc}"
            logging.warning("Attempt %d failed for %s: %s", attempt, prompt_id, last_error)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    return AssessmentResult(
        prompt_id=prompt_id,
        prompt=prompt_text,
        difficulty=None,
        rationale="",
        error=last_error,
    )


def load_prompts(input_dir: Path) -> list[tuple[str, str]]:
    """Load all `*.txt` files. The filename stem is the prompt_id."""
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    prompts: list[tuple[str, str]] = []
    for path in sorted(input_dir.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            logging.warning("Skipping empty file: %s", path.name)
            continue
        prompts.append((path.stem, text))
    return prompts


def write_csv(output_path: Path, results: list[AssessmentResult]) -> None:
    """Write results to CSV, sorted by prompt_id."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_id", "prompt", "difficulty", "difficulty_label", "rationale", "error"])
        for r in sorted(results, key=lambda x: x.prompt_id):
            label = DIFFICULTY_LABELS.get(r.difficulty, "") if r.difficulty is not None else ""
            writer.writerow([
                r.prompt_id,
                r.prompt,
                r.difficulty if r.difficulty is not None else "",
                label,
                r.rationale,
                r.error,
            ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Assess prompt difficulty via the OpenAI API.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing [prompt_id].txt files.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Path for the output CSV file.")
    parser.add_argument("--model", type=str, default="gpt-5.4", help="Model name to use (default: gpt-5.4).")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent API calls (default: 4).")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key. Defaults to the OPENAI_API_KEY env var.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Provide an API key via --api-key or the OPENAI_API_KEY env var.", file=sys.stderr)
        return 1

    client = OpenAI(api_key=api_key)

    prompts = load_prompts(args.input_dir)
    if not prompts:
        logging.warning("No prompts found in %s", args.input_dir)
        write_csv(args.output_csv, [])
        return 0

    logging.info("Loaded %d prompt(s). Assessing with model=%s ...", len(prompts), args.model)

    results: list[AssessmentResult] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_to_id = {
            pool.submit(assess_one_prompt, client, args.model, pid, text): pid
            for pid, text in prompts
        }
        for fut in as_completed(future_to_id):
            r = fut.result()
            if r.error:
                logging.error("Failed: %s -> %s", r.prompt_id, r.error)
            else:
                logging.info("Done: %s -> %s (%s)", r.prompt_id, r.difficulty, DIFFICULTY_LABELS[r.difficulty])
            results.append(r)

    write_csv(args.output_csv, results)
    succeeded = sum(1 for r in results if r.difficulty is not None)
    logging.info("Wrote %d row(s) to %s (%d succeeded, %d failed).",
                 len(results), args.output_csv, succeeded, len(results) - succeeded)
    return 0


if __name__ == "__main__":
    sys.exit(main())