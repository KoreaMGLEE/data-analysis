"""
스크립트 2: 정의 B (학습 진행 기반 easiness, learning speed) 점수 계산

Base 모델로 loss를 계산한 후, 짧게 학습시킨 모델에서 다시 loss를 계산하여
loss 감소량(learning speed)을 측정합니다.

Usage:
    python preliminary/script2_definition_b.py \
        --base_model_name EleutherAI/pythia-160m \
        --output_dir ./results/definition_b \
        --training_epochs 1 \
        --training_lr 5e-5 \
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

# Project root 계산 (preliminary/script2_definition_b.py -> true_diff/)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.train import train_model, evaluate_examples
from data.dataloader import load_mnli_raw


def main():
    parser = argparse.ArgumentParser(description="Definition B: Learning speed easiness")
    parser.add_argument("--base_model_name", type=str, default="EleutherAI/pythia-160m",
                        help="Base model name (before training)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for scores and easy sets")
    parser.add_argument("--training_epochs", type=int, default=1,
                        help="Number of epochs for short training")
    parser.add_argument("--training_lr", type=float, default=5e-5,
                        help="Learning rate for short training")
    parser.add_argument("--training_batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--top_k_percent", type=float, nargs="+", default=[5, 10, 20, 40],
                        help="Top k percentages for easy sets")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda if available)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and use existing checkpoint")
    parser.add_argument("--trained_checkpoint", type=str, default=None,
                        help="Path to already trained checkpoint (if skip_training)")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("Definition B: Learning Speed Easiness")
    print("="*70)
    print(f"Base model: {args.base_model_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {device}")
    
    # Load MNLI train data
    print(f"\n[1/5] Loading MNLI training data...")
    train_dataset = load_mnli_raw(split="train", limit=None)
    train_dataset = train_dataset.map(
        lambda x, idx: {"example_id": idx},
        with_indices=True
    )
    print(f"  ✓ Loaded {len(train_dataset)} examples")
    
    # Step 1: Calculate loss with base model
    print(f"\n[2/5] Calculating loss with base model...")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name, trust_remote_code=True)
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=True)
    
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    
    base_model.to(device)
    base_model.eval()
    
    before_results = evaluate_examples(
        model=base_model,
        tokenizer=base_tokenizer,
        dataset=train_dataset,
        device=device,
        batch_size=args.eval_batch_size,
        max_examples=None,
    )
    
    # Create loss_before map
    loss_before_map = {r["example_id"]: r["nll"] for r in before_results}
    print(f"  ✓ Calculated loss for {len(loss_before_map)} examples")
    
    # Step 2: Train model briefly
    if args.skip_training:
        if args.trained_checkpoint is None:
            raise ValueError("--trained_checkpoint required when --skip_training is used")
        trained_checkpoint = args.trained_checkpoint
        print(f"\n[3/5] Skipping training, using existing checkpoint: {trained_checkpoint}")
    else:
        print(f"\n[3/5] Training model for {args.training_epochs} epoch(s)...")
        training_output_dir = os.path.join(args.output_dir, "trained_checkpoint")
        os.makedirs(training_output_dir, exist_ok=True)
        
        _, _, _, _ = train_model(
            model_name=args.base_model_name,
            output_dir=training_output_dir,
            num_train_epochs=args.training_epochs,
            batch_size=args.training_batch_size,
            learning_rate=args.training_lr,
            eval_strategy="epoch",
            num_proc=16,
            tokenize_batch_size=1000,
            save_total_limit=1,
        )
        trained_checkpoint = training_output_dir
        print(f"  ✓ Training completed, checkpoint: {trained_checkpoint}")
    
    # Step 3: Calculate loss with trained model
    print(f"\n[4/5] Calculating loss with trained model...")
    trained_model = AutoModelForCausalLM.from_pretrained(trained_checkpoint, trust_remote_code=True)
    trained_tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint, trust_remote_code=True)
    
    if trained_tokenizer.pad_token is None:
        trained_tokenizer.pad_token = trained_tokenizer.eos_token
        trained_tokenizer.pad_token_id = trained_tokenizer.eos_token_id
    
    trained_model.to(device)
    trained_model.eval()
    
    after_results = evaluate_examples(
        model=trained_model,
        tokenizer=trained_tokenizer,
        dataset=train_dataset,
        device=device,
        batch_size=args.eval_batch_size,
        max_examples=None,
    )
    
    # Create loss_after map
    loss_after_map = {r["example_id"]: r["nll"] for r in after_results}
    print(f"  ✓ Calculated loss for {len(loss_after_map)} examples")
    
    # Step 4: Calculate learning speed (delta_loss = loss_before - loss_after)
    print(f"\n[5/5] Calculating learning speed (delta_loss)...")
    scores_data = []
    for example_id in loss_before_map.keys():
        if example_id not in loss_after_map:
            continue
        
        loss_before = loss_before_map[example_id]
        loss_after = loss_after_map[example_id]
        delta_loss = loss_before - loss_after  # Positive = easier to learn
        
        # Find corresponding result for metadata
        after_result = next(r for r in after_results if r["example_id"] == example_id)
        
        scores_data.append({
            "example_id": example_id,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "delta_loss": delta_loss,  # Learning speed: larger = easier
            "true_label": after_result["true_label"],
            "predicted_label": after_result["predicted_label"],
            "correct": after_result["correct"],
        })
    
    # Sort by delta_loss (larger delta_loss = easier to learn)
    scores_data.sort(key=lambda x: x["delta_loss"], reverse=True)
    
    # Save scores
    scores_file = os.path.join(args.output_dir, "definition_b_scores.jsonl")
    print(f"\nSaving scores to {scores_file}...")
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
        "base_model_name": args.base_model_name,
        "trained_checkpoint": trained_checkpoint,
        "training_epochs": args.training_epochs,
        "training_lr": args.training_lr,
        "total_examples": total_examples,
        "top_k_percent": args.top_k_percent,
        "easy_set_sizes": {k: len(v) for k, v in easy_sets.items()},
        "seed": args.seed,
        "scoring_method": "learning_speed (delta_loss = loss_before - loss_after)",
        "note": "Larger delta_loss = easier to learn (faster learning)"
    }
    metadata_file = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved metadata to {metadata_file}")
    
    # Print statistics
    delta_losses = [item["delta_loss"] for item in scores_data]
    print(f"\nDelta Loss Statistics:")
    print(f"  Min: {min(delta_losses):.4f}")
    print(f"  Max: {max(delta_losses):.4f}")
    print(f"  Mean: {np.mean(delta_losses):.4f}")
    print(f"  Median: {np.median(delta_losses):.4f}")
    print(f"  Std: {np.std(delta_losses):.4f}")
    
    print("\n" + "="*70)
    print("Definition B scoring completed!")
    print("="*70)


if __name__ == "__main__":
    main()

