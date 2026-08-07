"""Run the fixed eval cases against the RAG pipeline and check expectations.

Usage: python eval/run_eval.py
Exit code 0 if every case passes, 1 otherwise.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag_lib import answer, get_embedder, load_index  # noqa: E402

CASES_PATH = Path(__file__).resolve().parent / "cases.json"


def check_case(case: dict, result: dict) -> list:
    """Return a list of failure reasons; empty list means the case passed."""
    failures = []
    refused = result["refused"]
    answer_text = result["answer"]

    if refused != case["expect_refusal"]:
        failures.append(
            f"expected refused={case['expect_refusal']}, got refused={refused}"
        )

    if not refused:
        if "must_cite_display_id" in case:
            cited = [c["display_id"] for c, _ in result["results"][:1]]
            if case["must_cite_display_id"] not in cited:
                failures.append(
                    f"expected top citation NCD {case['must_cite_display_id']}, "
                    f"got {cited}"
                )
        if "must_contain_any" in case:
            haystack = answer_text.lower()
            if not any(s.lower() in haystack for s in case["must_contain_any"]):
                failures.append(
                    f"answer text did not contain any of {case['must_contain_any']!r}"
                )
    else:
        # A refusal must actually say so, not silently omit an answer.
        if "can't determine" not in answer_text and "cannot determine" not in answer_text:
            failures.append("refused=True but answer text doesn't read as a refusal")

    return failures


def main():
    chunks, embeddings = load_index()
    model = get_embedder()
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    n_pass = 0
    for case in cases:
        result = answer(case["question"], chunks, embeddings, model=model)
        failures = check_case(case, result)

        status = "PASS" if not failures else "FAIL"
        if not failures:
            n_pass += 1

        print(f"[{status}] {case['id']}  ({case['type']})")
        print(f"        Q: {case['question']}")
        top = result["results"][0]
        print(f"        top match: {top[1]:.3f}  {top[0]['display_id']}  refused={result['refused']}")
        if failures:
            for f in failures:
                print(f"        FAILURE: {f}")
        print()

    print(f"{n_pass}/{len(cases)} cases passed")
    if n_pass != len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
