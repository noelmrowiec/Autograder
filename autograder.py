#!/usr/bin/env python3
# Note: code written with the help of Open AI' ChatGPT o3 model
"""
Auto-generate grading prompts and send them to OpenAI's Chat Completions API.

This script is a command-line program  
It:

1. Reads a *student answers* JSON file and a *rubric* JSON file.
2. Builds, for **each** sub-part that needs grading, a rich prompt containing:
   • the overall question statement (“Given”),  
   • context from **previous** parts (question + student answer + rubric),  
   • the *current* part's question and student answer,  
   • the *solution rubric* (split into sub-parts with marking codes/notes).  
3. Sends every prompt to the OpenAI API (parallelized) and prints the raw JSON
   responses.  You can pipe them to a file for post-processing.

Usage
-----
$ export OPENAI_API_KEY="sk-..."
$ python autograder_openai.py mt1_student_solution.json mt1_rubric_solution.json \
          --model gpt-4o-mini

Dependencies
------------
`pip install openai json5`   (json5 only required if your rubric uses JSON5)

JSON file schema (simplified)
-----------------------------
{
  "examTitle": "Midterm 1",
  "questions": [
    {
      "questionNumber": 128,
      "problemStatement": "Let …",
      "subproblems": [
        {
          "partIdentifier": "A",
          "problemStatement": "Show that …",
          "solution": "…"
          // Optional extra fields coming from the rubric author:
          "markingCode": "M1",
          "notes": ""
        },
        …
      ]
    },
    …
  ]
}
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Any, Tuple

import openai

import utils

###############################################################################
# Prompt header & footer (abridged from the original notebook)
###############################################################################

PROMPT_HEADER = textwrap.dedent(
    """\
    You are a JSON-generating IB Math exam grader.
    Your response **must** be valid, well-formed JSON, with *no* additional text
    before or after the JSON object.

    Here is the context:
    {
    """)

PROMPT_FOOTER = textwrap.dedent(
    """\
    }

    In the above context JSON, there is:
    • **Given** - the statement common to all parts of this question.  
    • **Context from previous parts** - what the student previously answered.  
    • **Current part** - the question & student answer you are grading now.  
    • **Rubric solution** - the solution broken into sub-parts, each with an
      `answer`, `markingCode`, and optional `notes`.

    Please award each markingCode as `"yes"` or `"no"` and justify your
    decision concisely but rigorously.

    Respond with JSON of the following form **and nothing else**:

      {
        "grade": [
          {
            "markingCode_component": "M1",
            "awarded": "yes",
            "reason": "…"
          },
          …
        ]
      }
    """)


###############################################################################
# Utility functions
###############################################################################

def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON or JSON5 file (the latter if the file contains comments)."""
    try:
        import json5  # Optional dependency
        loader = json5.load
    except ImportError:
        loader = json.load

    try:
        with path.open('r', encoding='utf-8') as fh:
            return loader(fh)
    except Exception as exc:
        print(f"[ERROR] Cannot load {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def build_prompts(student: Dict[str, Any], rubric: Dict[str, Any]) -> List[str]:
    """Create one prompt per part using the notebook’s logic."""
    prompts: List[str] = []

    # Iterate over questions (assumes same ordering in both files)
    for q_rub, q_stu in zip(rubric['questions'], student['questions']):
        given_text = q_rub['problemStatement']
        rubric_subparts = q_rub.get('subproblems', [])
        student_subparts = q_stu.get('subproblems', [])

        # Sanity: same number of parts
        if len(rubric_subparts) != len(student_subparts):
            print(f"[WARN] Question {q_rub['questionNumber']}: mismatch in sub‑parts", file=sys.stderr)

        previous_parts_context: List[Dict[str, str]] = []

        for idx, (rub_sub, stu_sub) in enumerate(zip(rubric_subparts, student_subparts), start=1):
            # Assemble JSON *context* block:
            current_part = {
                "number_part": f"{q_rub['questionNumber']}{rub_sub['partIdentifier']}",
                "part_question": rub_sub['problemStatement'],
                "student_answer": stu_sub.get('solution', ''),
                "rubric_answer": rub_sub['solution'],
            }

            # Build 'solution rubric' for **all** subparts (not just current one)
            # This matches the original notebook behaviour.
            solution_rubric = []
            for sub in rubric_subparts:
                solution_rubric.append({
                    "answer": sub['solution'],
                    "markingCode": sub.get('markingCode', ''),   # may be empty
                    "notes": sub.get('notes', '')
                })

            context_json = {
                "given": given_text,
                "previous_parts_context": previous_parts_context,
                "current_part": current_part,
                "solution_rubric": solution_rubric,
            }

            prompt_body = json.dumps(context_json, ensure_ascii=False, indent=2)
            prompt = PROMPT_HEADER + prompt_body + PROMPT_FOOTER
            prompts.append(prompt)

            # Update previous_parts_context for *next* iteration
            previous_parts_context.append({
                "number_part": current_part['number_part'],
                "part_question": current_part['part_question'],
                "student_answer": current_part['student_answer'],
                "rubric_answer": current_part['rubric_answer'],
            })

    return prompts


def call_openai_api(prompt_text: str, model: str, temperature: float = 0.0) -> str:
    """Single OpenAI chat completion call (synchronous)."""
    # Using OpenAI Python Library v1.x
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt_text}]}],
        temperature=temperature,
        response_format={"type": "json_object"},
        # You may tune the following as needed:
        max_tokens=1024,
    )
    return response.choices[0].message.content


###############################################################################
# Main
###############################################################################

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Grade IB Math answers using the OpenAI API.")
    parser.add_argument("student_json", type=Path, help="Path to the student's answers JSON")
    parser.add_argument("rubric_json", type=Path, help="Path to the rubric answers JSON")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name (default: gpt‑4o‑mini)")
    parser.add_argument("--threads", type=int, default=4, help="Concurrent API calls (default: 4)")
    args = parser.parse_args(argv)

    # ----------------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------------
    student_data = load_json(args.student_json)
    rubric_data = load_json(args.rubric_json)

    # ----------------------------------------------------------------------
    # Build prompts
    # ----------------------------------------------------------------------
    prompts = build_prompts(student_data, rubric_data)
    print(f"Built {len(prompts)} prompts.  Submitting to OpenAI…", file=sys.stderr)

    # ----------------------------------------------------------------------
    # Call the API (parallel)
    # ----------------------------------------------------------------------

    results: List[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = [pool.submit(call_openai_api, prompt, args.model) for prompt in prompts]
        for fut in concurrent.futures.as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as exc:
                results.append(json.dumps({"error": str(exc)}))

    # ----------------------------------------------------------------------
    # Output
    # ----------------------------------------------------------------------
    for idx, (prompt, response) in enumerate(zip(prompts, results), start=1):
        print(f"### Response {idx} ###{response}")

    print(f"Completed {len(results)} API calls.", file=sys.stderr)


if __name__ == "__main__":
    main()
