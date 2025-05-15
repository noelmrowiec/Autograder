#!/usr/bin/env python3
# Note: code written with the help of Open AI' ChatGPT o3 model
"""
extract_exam.py

This program will read exam PDFs and output a JSON file. 

The program relies on split_exam_to_problems.py for splitting the exam into images.
This file will read every JPG in ./problem_images (which is created by this program 
and then deleted), then ask Open AI model to perform OCR + structure the
contents and finally compile the whole exam into one JSON file.

Requirements:
  pip install --upgrade "openai>=1.15" pillow tqdm python-dotenv
  export OPENAI_API_KEY="sk-..."

Usage:
  python extract_exam.py --title "Midterm 1" --out exam.json

The program will prompt the user for a directory containing the PDFs of the exams
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from openai import AsyncOpenAI         # openai-python ≥ 1.15
from PIL import Image
from tqdm.asyncio import tqdm_asyncio

import split_exam_to_problems as exam_processor
from utils import load_api_key

# ────────────────────────── constants ──────────────────────────
MODEL_ID         = "o4-mini"
PAR_CONCURRENCY  = 8           # how many requests at once
MAX_COMP_TOKENS  = 4096        # per-image completion budget

# global handles set in async_main()
client: AsyncOpenAI
sem: asyncio.Semaphore

# ---------- helper functions -------------------------------------------------

def _image_to_b64(path: Path) -> str:
    """Load an image from disk and return base64-encoded PNG data string."""
    buf = BytesIO()
    Image.open(path).convert("RGB").save(buf, format="PNG")
    import base64

    return base64.b64encode(buf.getvalue()).decode()

def _build_prompt() -> str:
    """One reusable text prompt sent with every image."""
    return (
        "You will receive exactly one math problem as an image. "
        "Perform OCR on the text in the problem. Read the printed " 
        "problem statement and a hand-written solution from the student. "
        "Detect whether the problem contains sub-problems such as (a), (b), (c)…\n\n"
        "• If sub-problems exist, return **one** JSON object with the schema:\n"
        "{\n"
        '  "questionNumber": <int>,               # leave blank if unknown\n'
        '  "problemStatement": "<LaTeX>",\n'
        '  "subproblems": [\n'
        "    {\n"
        '      "partIdentifier": "A",            # capital letter\n'
        '      "problemStatement": "<LaTeX>",\n'
        '      "solution": "<LaTeX>"\n'
        "    }, ...\n"
        "  ]\n"
        "}\n\n"
        "• If there are NO sub-problems, return **one** JSON object with:\n"
        "{\n"
        '  "questionNumber": <int>,\n'
        '  "problemStatement": "<LaTeX>",\n'
        '  "solution": "<LaTeX>"\n'
        "}\n\n"
        "Rules:\n"
        "  - Use valid JSON only (no comments, no trailing commas).\n"
        "  - Escape LaTeX backslashes properly (e.g. \"\\\\frac{1}{2}\").\n"
        "  - Do not wrap the JSON in markdown fences.\n"
        "  - Do not make anything up; perform OCR.\n"
    )


def _error_question(qnum: int, reason: str) -> dict:
    """
    Return a minimal question-shaped JSON object that flags an OCR failure.
    """
    return {
        "questionNumber": qnum,
        "problemStatement": f"**OCR ERROR:** {reason}",
        "solution": "",
    }

async def _ocr_one(path: Path, qnum: int) -> Dict[str, Any]:
    """
    Send one image to o4-mini with limited parallelism.
    Falls back to an error JSON block if the reply is empty or malformed.
    """
    async with sem:  # limit concurrency
        b64 = _image_to_b64(path)
        data_uri = f"data:image/png;base64,{b64}"

        resp = await client.chat.completions.create(
            model=MODEL_ID,
            reasoning_effort="low",
            max_completion_tokens=MAX_COMP_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _build_prompt()},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "OCR this image and return the JSON object only.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri, "detail": "low"},
                        },
                    ],
                },
            ],
        )

    finish_reason = resp.choices[0].finish_reason
    raw = resp.choices[0].message.content or ""

    if not raw.strip():
        return _error_question(
            qnum, f"assistant returned empty content (finish_reason='{finish_reason}')"
        )

    try:
        obj = json.loads(raw)
        #obj["questionNumber"] = qnum  # ensure field present
        return obj
    except json.JSONDecodeError as exc:
        return _error_question(qnum, f"JSON parse error: {exc}")
    

# ────────────────────────── async driver ──────────────────────────
async def async_main(title: str, out_file: str) -> None:
    img_root = Path(exam_processor.get_images_folder())
    images: List[Path] = sorted(img_root.rglob("*.jpg"))
    if not images:
        sys.exit(f"❌  No JPGs found under {img_root}")

    global client, sem
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(PAR_CONCURRENCY)

    tasks = [_ocr_one(p, i) for i, p in enumerate(images, 1)]
    questions = await tqdm_asyncio.gather(*tasks)  # progress bar keeps order

    exam = {"examTitle": title, "questions": questions}
    Path(out_file).write_text(json.dumps(exam, indent=2, ensure_ascii=False))
    print(f"✅  Wrote {len(questions)} questions to {out_file}")

    
def main(argv: List[str] | None = None) -> None:
    # build images from PDFs
    exam_processor.process_all_exams()
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="Exam 1")
    parser.add_argument("--out", default="exam.json")
    args = parser.parse_args(argv)
    asyncio.run(async_main(args.title, args.out))
    exam_processor.delete_problem_images()

if __name__ == "__main__":
    main()
