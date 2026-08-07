"""Empirically calibrate REFUSAL_THRESHOLD in src/rag_lib.py.

Runs a fixed set of on-topic probe questions (one or more per indexed
document) and clearly off-topic probe questions against the current local
index, and reports the top-1 cosine similarity for each. The threshold in
rag_lib.py should sit in the gap between the lowest on-topic score and the
highest off-topic score.

Re-run this whenever the corpus changes (documents added/removed) -- the
gap can shift, and REFUSAL_THRESHOLD should be re-checked against it.

Usage: python scripts/calibrate_threshold.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag_lib import get_embedder, load_index, retrieve  # noqa: E402

ON_TOPIC_PROBES = [
    "Is a screening colonoscopy covered for a 55 year old at average risk for colorectal cancer?",
    "Is a fecal occult blood test covered once a year for a 50 year old?",
    "Is cardiac rehab covered for a patient with heart failure and an ejection fraction of 40 percent?",
    "Is a PET scan covered for initial staging of newly diagnosed breast cancer?",
    "Is CPAP covered for a patient newly diagnosed with obstructive sleep apnea?",
]

OFF_TOPIC_PROBES = [
    "Is a total knee replacement covered for a 70 year old with severe osteoarthritis?",
    "Does Medicare cover acupuncture for chronic lower back pain?",
    "Is bariatric weight loss surgery covered for a patient with a BMI of 42?",
]


def main():
    chunks, embeddings = load_index()
    model = get_embedder()

    def top1(question):
        return retrieve(question, chunks, embeddings, model=model, top_k=1)[0][1]

    on_scores = [top1(q) for q in ON_TOPIC_PROBES]
    off_scores = [top1(q) for q in OFF_TOPIC_PROBES]

    print("On-topic probes:")
    for q, s in zip(ON_TOPIC_PROBES, on_scores):
        print(f"  {s:.3f}  {q}")
    print("\nOff-topic probes:")
    for q, s in zip(OFF_TOPIC_PROBES, off_scores):
        print(f"  {s:.3f}  {q}")

    print(f"\nOn-topic range:  {min(on_scores):.3f} - {max(on_scores):.3f}")
    print(f"Off-topic range: {min(off_scores):.3f} - {max(off_scores):.3f}")
    gap_low, gap_high = max(off_scores), min(on_scores)
    if gap_low < gap_high:
        print(f"Gap: ({gap_low:.3f}, {gap_high:.3f}) -- REFUSAL_THRESHOLD should sit in here")
    else:
        print("WARNING: on-topic and off-topic ranges overlap -- no clean threshold exists")


if __name__ == "__main__":
    main()
