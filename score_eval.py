import argparse
import json
import os
import re
from statistics import mean
from typing import Dict, List


def read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_keywords(reference: str) -> List[str]:
    if not reference:
        return []
    parts = re.split(r"[，,。；;、\s]+", reference)
    keywords = [x.strip() for x in parts if len(x.strip()) >= 2]
    return keywords[:6]


def repetition_ratio(text: str, n: int = 3) -> float:
    if len(text) <= n:
        return 0.0
    grams = [text[i : i + n] for i in range(0, len(text) - n + 1)]
    if not grams:
        return 0.0
    unique = len(set(grams))
    return 1.0 - (unique / len(grams))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def score_one(row: Dict) -> Dict:
    pred = str(row.get("prediction", "")).strip()
    ref = str(row.get("reference", "")).strip()
    latency = float(row.get("latency_sec", 0.0))
    tokens = int(row.get("gen_tokens", 0))

    char_len = len(pred)
    rep = repetition_ratio(pred, n=3)
    keywords = extract_keywords(ref)
    hit = 0
    if keywords:
        hit = sum(1 for k in keywords if k in pred)
        keyword_hit_rate = hit / len(keywords)
    else:
        keyword_hit_rate = 0.0

    # Length score: favor practical responses, avoid too short/too long
    if char_len < 12:
        length_score = 0.2
    elif char_len <= 280:
        length_score = 1.0
    elif char_len <= 500:
        length_score = 0.6
    else:
        length_score = 0.2

    repetition_penalty = clamp(1.0 - rep, 0.0, 1.0)
    speed = (tokens / latency) if latency > 0 else 0.0
    speed_score = clamp(speed / 30.0, 0.0, 1.0)

    final_score = (
        0.45 * length_score
        + 0.30 * repetition_penalty
        + 0.15 * keyword_hit_rate
        + 0.10 * speed_score
    )
    final_score = round(final_score * 100, 2)

    return {
        "id": row.get("id", ""),
        "index": row.get("index", 0),
        "char_len": char_len,
        "gen_tokens": tokens,
        "latency_sec": round(latency, 3),
        "tok_per_sec": round(speed, 3),
        "repetition_ratio": round(rep, 4),
        "keyword_count": len(keywords),
        "keyword_hit": hit,
        "keyword_hit_rate": round(keyword_hit_rate, 4),
        "score": final_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score eval_results.jsonl with simple quality heuristics")
    parser.add_argument("--in_file", type=str, default="eval_results.jsonl")
    parser.add_argument("--out_file", type=str, default="eval_scores.jsonl")
    parser.add_argument("--summary_file", type=str, default="eval_summary.json")
    args = parser.parse_args()

    if not os.path.exists(args.in_file):
        raise FileNotFoundError(f"Input file not found: {args.in_file}")

    rows = read_jsonl(args.in_file)
    if not rows:
        raise ValueError("No rows found in input jsonl")

    scored = [score_one(r) for r in rows]
    scored_sorted = sorted(scored, key=lambda x: x["score"], reverse=True)

    with open(args.out_file, "w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    scores = [r["score"] for r in scored]
    tps = [r["tok_per_sec"] for r in scored]
    reps = [r["repetition_ratio"] for r in scored]
    summary = {
        "samples": len(scored),
        "avg_score": round(mean(scores), 2),
        "min_score": round(min(scores), 2),
        "max_score": round(max(scores), 2),
        "avg_tok_per_sec": round(mean(tps), 3),
        "avg_repetition_ratio": round(mean(reps), 4),
        "top3": scored_sorted[:3],
        "bottom3": scored_sorted[-3:],
    }

    with open(args.summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Scored {len(scored)} samples")
    print(f"avg_score={summary['avg_score']} avg_tok_per_sec={summary['avg_tok_per_sec']}")
    print(f"Saved: {args.out_file}")
    print(f"Saved: {args.summary_file}")


if __name__ == "__main__":
    main()
