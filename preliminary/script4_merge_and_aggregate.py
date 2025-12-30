"""
스크립트 4: 결과 병합 및 Bias Rate 집계

정의 A, 정의 B 점수 파일과 hypothesis-only bias 태그를 병합하여
각 easy set의 bias rate를 계산합니다.

Usage:
    python preliminary/script4_merge_and_aggregate.py \
        --definition_a_scores ./results/definition_a/definition_a_scores.jsonl \
        --definition_b_scores ./results/definition_b/definition_b_scores.jsonl \
        --bias_tags ./results/hypothesis_bias_tags.jsonl \
        --output_dir ./results/merged
"""

import os
import sys
import json
import argparse
import pandas as pd
from collections import defaultdict

# Project root 계산 (preliminary/script4_merge_and_aggregate.py -> true_diff/)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)


def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_json(file_path):
    """Load JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Merge scores and calculate bias rates")
    parser.add_argument("--definition_a_scores", type=str, required=True,
                        help="Definition A scores JSONL file")
    parser.add_argument("--definition_b_scores", type=str, required=True,
                        help="Definition B scores JSONL file")
    parser.add_argument("--bias_tags", type=str, required=True,
                        help="Hypothesis-only bias tags JSONL file")
    parser.add_argument("--definition_a_dir", type=str, default=None,
                        help="Definition A output directory (for easy sets)")
    parser.add_argument("--definition_b_dir", type=str, default=None,
                        help="Definition B output directory (for easy sets)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for merged results")
    parser.add_argument("--top_k_percent", type=float, nargs="+", default=[5, 10, 20, 40],
                        help="Top k percentages for easy sets")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("Merge and Aggregate Results")
    print("="*70)
    
    # Load scores and bias tags
    print(f"\n[1/4] Loading scores and bias tags...")
    def_a_scores = load_jsonl(args.definition_a_scores)
    def_b_scores = load_jsonl(args.definition_b_scores)
    bias_tags = load_jsonl(args.bias_tags)
    
    print(f"  Definition A scores: {len(def_a_scores)} examples")
    print(f"  Definition B scores: {len(def_b_scores)} examples")
    print(f"  Bias tags: {len(bias_tags)} examples")
    
    # Create bias tag map
    bias_map = {item["example_id"]: item["hypothesis_only_bias"] for item in bias_tags}
    
    # Create definition A score map
    def_a_map = {item["example_id"]: item["snapshot_loss"] for item in def_a_scores}
    
    # Create definition B score map
    def_b_map = {item["example_id"]: item["delta_loss"] for item in def_b_scores}
    
    # Merge all data
    print(f"\n[2/4] Merging data...")
    merged_data = []
    all_example_ids = set(bias_map.keys())
    
    for example_id in all_example_ids:
        item = {
            "example_id": example_id,
            "hypothesis_only_bias": bias_map.get(example_id, 0),
            "definition_a_score": def_a_map.get(example_id, None),
            "definition_b_score": def_b_map.get(example_id, None),
        }
        merged_data.append(item)
    
    # Save merged data
    merged_file = os.path.join(args.output_dir, "merged_scores.jsonl")
    print(f"\n[3/4] Saving merged data to {merged_file}...")
    with open(merged_file, "w", encoding="utf-8") as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✓ Saved {len(merged_data)} merged examples")
    
    # Also save as CSV/Parquet for easier analysis
    df = pd.DataFrame(merged_data)
    csv_file = os.path.join(args.output_dir, "merged_scores.csv")
    df.to_csv(csv_file, index=False)
    print(f"  ✓ Saved CSV to {csv_file}")
    
    parquet_file = os.path.join(args.output_dir, "merged_scores.parquet")
    df.to_parquet(parquet_file, index=False)
    print(f"  ✓ Saved Parquet to {parquet_file}")
    
    # Load easy sets if directories provided
    print(f"\n[4/4] Calculating bias rates for easy sets...")
    
    bias_rates = {
        "definition_a": {},
        "definition_b": {},
    }
    
    if args.definition_a_dir:
        print(f"\n  Definition A easy sets:")
        for k_percent in args.top_k_percent:
            easy_set_file = os.path.join(args.definition_a_dir, f"easy_set_top_{int(k_percent)}_percent.json")
            if os.path.exists(easy_set_file):
                easy_ids = set(load_json(easy_set_file))
                bias_count = sum(1 for eid in easy_ids if bias_map.get(eid, 0) == 1)
                bias_rate = bias_count / len(easy_ids) if easy_ids else 0.0
                bias_rates["definition_a"][f"top_{int(k_percent)}"] = {
                    "easy_set_size": len(easy_ids),
                    "bias_count": bias_count,
                    "bias_rate": bias_rate,
                }
                print(f"    Top {k_percent}%: {bias_rate:.4f} ({bias_count}/{len(easy_ids)})")
    
    if args.definition_b_dir:
        print(f"\n  Definition B easy sets:")
        for k_percent in args.top_k_percent:
            easy_set_file = os.path.join(args.definition_b_dir, f"easy_set_top_{int(k_percent)}_percent.json")
            if os.path.exists(easy_set_file):
                easy_ids = set(load_json(easy_set_file))
                bias_count = sum(1 for eid in easy_ids if bias_map.get(eid, 0) == 1)
                bias_rate = bias_count / len(easy_ids) if easy_ids else 0.0
                bias_rates["definition_b"][f"top_{int(k_percent)}"] = {
                    "easy_set_size": len(easy_ids),
                    "bias_count": bias_count,
                    "bias_rate": bias_rate,
                }
                print(f"    Top {k_percent}%: {bias_rate:.4f} ({bias_count}/{len(easy_ids)})")
    
    # Calculate random baseline
    print(f"\n  Random baseline:")
    total_examples = len(all_example_ids)
    total_bias_count = sum(bias_map.values())
    overall_bias_rate = total_bias_count / total_examples if total_examples > 0 else 0.0
    print(f"    Overall bias rate: {overall_bias_rate:.4f} ({total_bias_count}/{total_examples})")
    
    random_baseline = {
        "overall_bias_rate": overall_bias_rate,
        "total_examples": total_examples,
        "total_bias_count": total_bias_count,
    }
    
    # Save bias rates
    results_file = os.path.join(args.output_dir, "bias_rates.json")
    output_data = {
        "bias_rates": bias_rates,
        "random_baseline": random_baseline,
        "top_k_percent": args.top_k_percent,
    }
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved bias rates to {results_file}")
    
    # Save summary table
    summary_rows = []
    for def_name, def_data in bias_rates.items():
        for k_name, k_data in def_data.items():
            summary_rows.append({
                "definition": def_name,
                "easy_set": k_name,
                "easy_set_size": k_data["easy_set_size"],
                "bias_count": k_data["bias_count"],
                "bias_rate": k_data["bias_rate"],
                "enrichment_ratio": k_data["bias_rate"] / overall_bias_rate if overall_bias_rate > 0 else 0.0,
            })
    
    # Add random baseline row
    summary_rows.append({
        "definition": "random",
        "easy_set": "all",
        "easy_set_size": total_examples,
        "bias_count": total_bias_count,
        "bias_rate": overall_bias_rate,
        "enrichment_ratio": 1.0,
    })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_file = os.path.join(args.output_dir, "bias_rates_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"  ✓ Saved summary table to {summary_file}")
    
    print("\n" + "="*70)
    print("Merge and aggregation completed!")
    print("="*70)


if __name__ == "__main__":
    main()

