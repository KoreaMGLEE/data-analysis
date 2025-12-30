"""
스크립트 1: 정의 A (단일 모델 스냅샷 기반 easiness) 점수 계산

고정된 체크포인트에서 MNLI train의 각 샘플에 대해 loss를 계산하고,
top-k% (k=5/10/20/40)로 easy set을 생성합니다.

Usage:
    python preliminary/script1_definition_a.py \
        --checkpoint_path ./checkpoints/pythia-160m-mnli \
        --output_dir ./results/definition_a \
        --top_k_percent 5 10 20 40 \
        --batch_size 32
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm

# Project root 계산 (preliminary/script1_definition_a.py -> true_diff/)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.train import evaluate_examples
from data.dataloader import load_mnli_raw


def main():
    parser = argparse.ArgumentParser(description="Definition A: Single model snapshot easiness")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to trained checkpoint (Pythia-160m trained for 1 epoch)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for scores and easy sets")
    parser.add_argument("--top_k_percent", type=float, nargs="+", default=[5, 10, 20, 40],
                        help="Top k percentages for easy sets")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda if available)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("Definition A: Single Model Snapshot Easiness")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {device}")
    
    # Load model
    print(f"\n[1/4] Loading model from {args.checkpoint_path}...")
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.to(device)
    model.eval()
    print("  ✓ Model loaded")
    
    # Load MNLI train data
    print(f"\n[2/4] Loading MNLI training data...")
    train_dataset = load_mnli_raw(split="train", limit=None)
    # Add example_id column
    train_dataset = train_dataset.map(
        lambda x, idx: {"example_id": idx},
        with_indices=True
    )
    print(f"  ✓ Loaded {len(train_dataset)} examples")
    
    # Calculate scores (loss/NLL) for all examples
    print(f"\n[3/4] Calculating snapshot loss for all examples...")
    results = evaluate_examples(
        model=model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        device=device,
        batch_size=args.batch_size,
        max_examples=None,
    )
    
    # Extract scores (NLL = loss)
    scores_data = []
    for r in results:
        scores_data.append({
            "example_id": r["example_id"],
            "snapshot_loss": r["nll"],
            "true_label": r["true_label"],
            "predicted_label": r["predicted_label"],
            "correct": r["correct"],
            "true_prob": r["true_prob"],
        })
    
    # Sort by loss (lower loss = easier)
    scores_data.sort(key=lambda x: x["snapshot_loss"])
    
    # Save scores
    scores_file = os.path.join(args.output_dir, "definition_a_scores.jsonl")
    print(f"\n[4/4] Saving scores to {scores_file}...")
    with open(scores_file, "w", encoding="utf-8") as f:
        for item in scores_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✓ Saved {len(scores_data)} scores")
    
    # Generate top-k% easy sets
    print(f"\nGenerating top-k% easy sets...")
    total_examples = len(scores_data)
    
    easy_sets = {}
    for k_percent in args.top_k_percent:
        k = int(total_examples * k_percent / 100)
        easy_ids = [item["example_id"] for item in scores_data[:k]]
        easy_sets[f"top_{k_percent}"] = easy_ids
        
        output_file = os.path.join(args.output_dir, f"easy_set_top_{int(k_percent)}_percent.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(easy_ids, f, indent=2)
        print(f"  ✓ Top {k_percent}%: {len(easy_ids)} examples -> {output_file}")
    
    # Save metadata
    metadata = {
        "checkpoint_path": args.checkpoint_path,
        "total_examples": total_examples,
        "top_k_percent": args.top_k_percent,
        "easy_set_sizes": {k: len(v) for k, v in easy_sets.items()},
        "seed": args.seed,
        "scoring_method": "snapshot_loss (NLL)",
        "note": "Lower loss = easier example"
    }
    metadata_file = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved metadata to {metadata_file}")
    
    # Print statistics
    losses = [item["snapshot_loss"] for item in scores_data]
    print(f"\nLoss Statistics:")
    print(f"  Min: {min(losses):.4f}")
    print(f"  Max: {max(losses):.4f}")
    print(f"  Mean: {np.mean(losses):.4f}")
    print(f"  Median: {np.median(losses):.4f}")
    print(f"  Std: {np.std(losses):.4f}")
    
    print("\n" + "="*70)
    print("Definition A scoring completed!")
    print("="*70)


if __name__ == "__main__":
    main()

