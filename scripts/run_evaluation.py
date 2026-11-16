"""
run_evaluation.py - Run the full NDCG@5 experiment from the command line.

Usage (from project root):
    python scripts/run_evaluation.py            # uses LLM if GROQ_API_KEY is set
    python scripts/run_evaluation.py --offline  # rule-based extraction only

Writes results/ndcg_summary.csv, results/ndcg_per_job.csv and results/ndcg_chart.png
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.evaluate import evaluate_methods, load_human_rankings
from src.extract import extract_cached
from src.io_utils import load_folder
from src.ranking import CandidateRanker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="skip the LLM, use rule-based extraction")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    use_llm = not args.offline

    print("1/3 Extracting resumes and job descriptions ...")
    resumes = {k: extract_cached("resumes", k, t, use_llm) for k, t in load_folder(ROOT / "data/resumes").items()}
    jds = {k: extract_cached("jds", k, t, use_llm) for k, t in load_folder(ROOT / "data/jds").items()}
    print(f"    {len(resumes)} resumes, {len(jds)} job descriptions")

    print("2/3 Ranking with GloVe, SBERT and Hybrid ...")
    ranker = CandidateRanker(resumes)
    per_jd, summary = evaluate_methods(ranker, jds, load_human_rankings(), k=args.k)

    print("3/3 Saving results ...")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df_sum, df_jd = pd.DataFrame(summary), pd.DataFrame(per_jd)
    df_sum.to_csv(out / "ndcg_summary.csv", index=False)
    df_jd.to_csv(out / "ndcg_per_job.csv", index=False)

    pivot = df_jd.pivot(index="job", columns="method", values=f"NDCG@{args.k}")[["glove", "sbert", "hybrid"]]
    ax = pivot.plot(kind="bar", figsize=(9, 4.5), color=["#999999", "#4C72B0", "#C44E52"])
    ax.set_ylabel(f"NDCG@{args.k}")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"NDCG@{args.k} per job: static vs contextual embeddings")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out / "ndcg_chart.png", dpi=150)

    print("\n" + df_sum.to_string(index=False))
    print("\n" + df_jd.drop(columns=[f"top{args.k}"]).to_string(index=False))


if __name__ == "__main__":
    main()
