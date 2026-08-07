"""Run the fixed eval cases against the RAG pipeline and check expectations.

Usage:
  python eval/run_eval.py                                   # local index
  python eval/run_eval.py --remote https://claimsrag-chat.fly.dev  # deployed API

Exit code 0 if every case passes, 1 otherwise. Same check_case() logic runs
against either backend, so "does the live deployment behave the same as the
local index" is a real automated check, not a manually re-typed curl command.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag_lib import answer, get_embedder, load_index  # noqa: E402

CASES_PATH = Path(__file__).resolve().parent / "cases.json"

# Fly.io scale-to-zero: the deploy step's own health check can leave the
# machine idle long enough to stop again before this step's first request
# arrives, so the *first* request after a deploy can be a full cold start
# (image already warm, but firecracker boot + model load) -- seen taking
# >30s in practice. Wait for /health explicitly instead of hoping a single
# request's timeout covers it.
WAIT_FOR_READY_SECS = 90


def wait_for_ready(base_url: str, timeout: int = WAIT_FOR_READY_SECS) -> None:
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/health", timeout=10)
            if resp.ok and resp.json().get("status") == "ok":
                return
        except requests.RequestException as e:
            last_err = e
        time.sleep(3)
    raise SystemExit(f"{base_url} did not become ready within {timeout}s (last error: {last_err})")


def remote_answer(base_url: str, question: str) -> dict:
    """Call the deployed /api/ask endpoint and reshape its response into the
    same {refused, answer, results: [(chunk_dict, score), ...]} shape that
    the local rag_lib.answer() returns, so check_case() works unmodified."""
    resp = requests.post(f"{base_url}/api/ask", json={"question": question}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    results = [(c, c["similarity"]) for c in data["citations"]]
    return {"refused": data["refused"], "answer": data["answer"], "results": results}


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
                    f"expected top citation display_id={case['must_cite_display_id']!r}, "
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--remote",
        metavar="BASE_URL",
        help="Run against a deployed /api/ask endpoint instead of the local index, "
        "e.g. --remote https://claimsrag-chat.fly.dev",
    )
    args = parser.parse_args()

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    if args.remote:
        base_url = args.remote.rstrip("/")
        run_one = lambda q: remote_answer(base_url, q)  # noqa: E731
        print(f"Running against remote: {base_url}")
        print(f"Waiting for /health (up to {WAIT_FOR_READY_SECS}s -- Fly scale-to-zero cold start)...")
        wait_for_ready(base_url)
        print("Ready.\n")
    else:
        chunks, embeddings = load_index()
        model = get_embedder()
        run_one = lambda q: answer(q, chunks, embeddings, model=model)  # noqa: E731

    n_pass = 0
    for case in cases:
        result = run_one(case["question"])
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
