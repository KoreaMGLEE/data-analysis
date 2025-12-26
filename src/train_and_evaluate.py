"""
Train and Evaluate 3 Scenarios

3가지 시나리오로 모델을 훈련하고 평가합니다:
1. Full MNLI training → Evaluate on MNLI dev + HANS
2. True-easy (1 epoch) → Full MNLI → Evaluate on MNLI dev + HANS
3. True-easy → Easy → Hard (curriculum, no overlap) → Evaluate on MNLI dev + HANS

Usage:
    # Scenario 1 (default: Qwen/Qwen2.5-0.5B, GLUE hyperparameters)
    python src/train_and_evaluate.py \
        --scenario 1 \
        --output_dir ./checkpoints/scenario1_full

    # Scenario 2
    python src/train_and_evaluate.py \
        --scenario 2 \
        --true_easy_json /home/user3/data-analysis/true_easy_examples_conf0.4_drop0.1_loss0.5_stage2_curriculum_pythia-410m.json \
        --output_dir ./checkpoints/scenario2_true_easy_full

    # Scenario 3
    python src/train_and_evaluate.py \
        --scenario 3 \
        --true_easy_json /home/user3/data-analysis/true_easy_examples_conf0.4_drop0.1_loss0.5_stage2_curriculum_pythia-410m.json \
        --stage1_easy_json /home/user3/data-analysis/easy_examples_confidence_pythia-160m_0.8_3_5e-05.json \
        --output_dir ./checkpoints/scenario3_curriculum_3stage
"""

import os
import sys
import json
import argparse
import torch

# 상위 폴더를 path에 추가 (프로젝트 루트)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from train import train_model
from evaluate import evaluate_mnli_dev, evaluate_hans
from data.dataloader import load_mnli_raw
from transformers import AutoModelForCausalLM, AutoTokenizer


def scenario1_full_mnli(args):
    """
    Scenario 1: Full MNLI training
    """
    print("="*70)
    print("Scenario 1: Full MNLI Training")
    print("="*70)
    
    output_dir = os.path.join(args.output_dir, "full_mnli")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[Training] Training on full MNLI training data...")
    print(f"  Model: {args.model_name}")
    print(f"  Output: {output_dir}")
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"  Epochs: {args.num_epochs}, LR: {args.learning_rate}")
    print(f"  Batch size per device: {args.batch_size}, Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    
    model, tokenizer, trainer, _ = train_model(
        model_name=args.model_name,
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        num_proc=args.num_proc,
        tokenize_batch_size=args.tokenize_batch_size,
        save_total_limit=args.save_total_limit,
        train_raw_override=None,  # Full dataset
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    print(f"\n✓ Training completed")
    
    # Evaluation
    print(f"\n[Evaluation] Evaluating on MNLI dev and HANS...")
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    results = {}
    if args.eval_mnli_dev:
        mnli_result = evaluate_mnli_dev(model, tokenizer, device=device, batch_size=args.eval_batch_size)
        results["mnli_dev"] = {
            "accuracy": mnli_result["accuracy"],
            "correct": mnli_result["correct"],
            "total": mnli_result["total"],
        }
    
    if args.eval_hans:
        hans_result = evaluate_hans(model, tokenizer, device=device, batch_size=args.eval_batch_size)
        if hans_result:
            results["hans"] = {
                "accuracy": hans_result["accuracy"],
                "entailment_accuracy": hans_result["entailment_accuracy"],
                "non_entailment_accuracy": hans_result["non_entailment_accuracy"],
            }
    
    # Save results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved evaluation results to: {results_path}")
    
    return output_dir, results


def scenario2_true_easy_full(args):
    """
    Scenario 2: True-easy (1 epoch) → Full MNLI
    """
    print("="*70)
    print("Scenario 2: True-Easy → Full MNLI Training")
    print("="*70)
    
    # Load true_easy JSON
    print(f"\n[Step 1] Loading true-easy examples...")
    with open(args.true_easy_json, "r", encoding="utf-8") as f:
        true_easy_data = json.load(f)
    
    if not isinstance(true_easy_data, list):
        raise ValueError(f"Expected list in JSON, got {type(true_easy_data)}")
    
    # Extract example IDs
    true_easy_ids = set()
    for item in true_easy_data:
        example_id = item.get("example_id")
        if example_id is not None:
            true_easy_ids.add(example_id)
    
    print(f"  Found {len(true_easy_ids)} true-easy examples")
    
    # Load full training data
    print(f"\n[Step 2] Loading full MNLI training data...")
    full_train_raw = load_mnli_raw(split="train", limit=None)
    
    # Create true-easy set
    sorted_true_easy_ids = sorted(list(true_easy_ids))
    true_easy_set = full_train_raw.select(sorted_true_easy_ids)
    true_easy_set = true_easy_set.map(
        lambda x, idx: {"example_id": sorted_true_easy_ids[idx]},
        with_indices=True
    )
    
    print(f"  True-easy set size: {len(true_easy_set)}")
    
    # Phase 1: Train on true-easy (1 epoch)
    output_dir_phase1 = os.path.join(args.output_dir, "scenario2_phase1_true_easy")
    os.makedirs(output_dir_phase1, exist_ok=True)
    
    print(f"\n[Phase 1] Training on true-easy set (1 epoch)...")
    print(f"  Output: {output_dir_phase1}")
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"  Batch size per device: {args.batch_size}, Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    
    model, tokenizer, trainer, _ = train_model(
        model_name=args.model_name,
        output_dir=output_dir_phase1,
        num_train_epochs=1,  # 1 epoch only
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        num_proc=args.num_proc,
        tokenize_batch_size=args.tokenize_batch_size,
        save_total_limit=args.save_total_limit,
        train_raw_override=true_easy_set,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    print(f"  ✓ Phase 1 training completed")
    
    # Phase 2: Continue training on full MNLI
    output_dir_phase2 = os.path.join(args.output_dir, "scenario2_phase2_full")
    os.makedirs(output_dir_phase2, exist_ok=True)
    
    print(f"\n[Phase 2] Continuing training on full MNLI data...")
    print(f"  Output: {output_dir_phase2}")
    print(f"  Continuing from: {output_dir_phase1}")
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"  Epochs: {args.num_epochs}, LR: {args.learning_rate}")
    print(f"  Effective batch size: {effective_batch_size}")
    
    model, tokenizer, trainer, _ = train_model(
        model_name=output_dir_phase1,  # Continue from phase 1
        output_dir=output_dir_phase2,
        num_train_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        num_proc=args.num_proc,
        tokenize_batch_size=args.tokenize_batch_size,
        save_total_limit=args.save_total_limit,
        train_raw_override=None,  # Full dataset
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    print(f"  ✓ Phase 2 training completed")
    
    # Evaluation
    print(f"\n[Evaluation] Evaluating on MNLI dev and HANS...")
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    results = {}
    if args.eval_mnli_dev:
        mnli_result = evaluate_mnli_dev(model, tokenizer, device=device, batch_size=args.eval_batch_size)
        results["mnli_dev"] = {
            "accuracy": mnli_result["accuracy"],
            "correct": mnli_result["correct"],
            "total": mnli_result["total"],
        }
    
    if args.eval_hans:
        hans_result = evaluate_hans(model, tokenizer, device=device, batch_size=args.eval_batch_size)
        if hans_result:
            results["hans"] = {
                "accuracy": hans_result["accuracy"],
                "entailment_accuracy": hans_result["entailment_accuracy"],
                "non_entailment_accuracy": hans_result["non_entailment_accuracy"],
            }
    
    # Save results
    results_path = os.path.join(output_dir_phase2, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved evaluation results to: {results_path}")
    
    return output_dir_phase2, results


def scenario3_curriculum_3stage(args):
    """
    Scenario 3: True-easy → Easy → Hard (curriculum, no overlap)
    """
    print("="*70)
    print("Scenario 3: True-Easy → Easy → Hard (Curriculum, No Overlap)")
    print("="*70)
    
    # Load true_easy JSON
    print(f"\n[Step 1] Loading true-easy and stage1 easy examples...")
    with open(args.true_easy_json, "r", encoding="utf-8") as f:
        true_easy_data = json.load(f)
    
    with open(args.stage1_easy_json, "r", encoding="utf-8") as f:
        stage1_easy_data = json.load(f)
    
    # Extract example IDs
    true_easy_ids = set()
    for item in true_easy_data:
        example_id = item.get("example_id")
        if example_id is not None:
            true_easy_ids.add(example_id)
    
    stage1_easy_ids = set()
    for item in stage1_easy_data:
        if isinstance(item, dict):
            example_id = item.get("example_id")
        else:
            example_id = item
        if example_id is not None:
            stage1_easy_ids.add(example_id)
    
    print(f"  True-easy examples: {len(true_easy_ids)}")
    print(f"  Stage1 easy examples: {len(stage1_easy_ids)}")
    
    # Load full training data
    print(f"\n[Step 2] Loading full MNLI training data and splitting...")
    full_train_raw = load_mnli_raw(split="train", limit=None)
    
    # Create sets (no overlap)
    # True-easy set
    sorted_true_easy_ids = sorted(list(true_easy_ids))
    true_easy_set = full_train_raw.select(sorted_true_easy_ids)
    true_easy_set = true_easy_set.map(
        lambda x, idx: {"example_id": sorted_true_easy_ids[idx]},
        with_indices=True
    )
    
    # Easy set (stage1 easy - true_easy)
    easy_ids = sorted(list(stage1_easy_ids - true_easy_ids))
    easy_set = full_train_raw.select(easy_ids)
    easy_set = easy_set.map(
        lambda x, idx: {"example_id": easy_ids[idx]},
        with_indices=True
    )
    
    # Hard set (remaining)
    hard_ids = [i for i in range(len(full_train_raw)) if i not in stage1_easy_ids]
    hard_set = full_train_raw.select(hard_ids)
    hard_set = hard_set.map(
        lambda x, idx: {"example_id": hard_ids[idx]},
        with_indices=True
    )
    
    print(f"  True-easy set size: {len(true_easy_set)}")
    print(f"  Easy set size: {len(easy_set)}")
    print(f"  Hard set size: {len(hard_set)}")
    print(f"  Total: {len(true_easy_set) + len(easy_set) + len(hard_set)} (expected: {len(full_train_raw)})")
    
    # Phase 1: Train on true-easy
    output_dir_phase1 = os.path.join(args.output_dir, "scenario3_phase1_true_easy")
    os.makedirs(output_dir_phase1, exist_ok=True)
    
    print(f"\n[Phase 1] Training on true-easy set...")
    print(f"  Output: {output_dir_phase1}")
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"  Epochs: {args.num_epochs_easy}, LR: {args.lr_easy}")
    print(f"  Effective batch size: {effective_batch_size}")
    
    model, tokenizer, trainer, _ = train_model(
        model_name=args.model_name,
        output_dir=output_dir_phase1,
        num_train_epochs=args.num_epochs_easy,
        batch_size=args.batch_size,
        learning_rate=args.lr_easy,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        num_proc=args.num_proc,
        tokenize_batch_size=args.tokenize_batch_size,
        save_total_limit=args.save_total_limit,
        train_raw_override=true_easy_set,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    print(f"  ✓ Phase 1 training completed")
    
    # Phase 2: Continue on easy set
    output_dir_phase2 = os.path.join(args.output_dir, "scenario3_phase2_easy")
    os.makedirs(output_dir_phase2, exist_ok=True)
    
    print(f"\n[Phase 2] Continuing training on easy set...")
    print(f"  Output: {output_dir_phase2}")
    print(f"  Continuing from: {output_dir_phase1}")
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"  Epochs: {args.num_epochs_easy}, LR: {args.lr_easy}")
    print(f"  Effective batch size: {effective_batch_size}")
    
    model, tokenizer, trainer, _ = train_model(
        model_name=output_dir_phase1,
        output_dir=output_dir_phase2,
        num_train_epochs=args.num_epochs_easy,
        batch_size=args.batch_size,
        learning_rate=args.lr_easy,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        num_proc=args.num_proc,
        tokenize_batch_size=args.tokenize_batch_size,
        save_total_limit=args.save_total_limit,
        train_raw_override=easy_set,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    print(f"  ✓ Phase 2 training completed")
    
    # Phase 3: Continue on hard set
    output_dir_phase3 = os.path.join(args.output_dir, "scenario3_phase3_hard")
    os.makedirs(output_dir_phase3, exist_ok=True)
    
    print(f"\n[Phase 3] Continuing training on hard set...")
    print(f"  Output: {output_dir_phase3}")
    print(f"  Continuing from: {output_dir_phase2}")
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"  Epochs: {args.num_epochs_hard}, LR: {args.lr_hard}")
    print(f"  Effective batch size: {effective_batch_size}")
    
    model, tokenizer, trainer, _ = train_model(
        model_name=output_dir_phase2,
        output_dir=output_dir_phase3,
        num_train_epochs=args.num_epochs_hard,
        batch_size=args.batch_size,
        learning_rate=args.lr_hard,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        num_proc=args.num_proc,
        tokenize_batch_size=args.tokenize_batch_size,
        save_total_limit=args.save_total_limit,
        train_raw_override=hard_set,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    print(f"  ✓ Phase 3 training completed")
    
    # Evaluation
    print(f"\n[Evaluation] Evaluating on MNLI dev and HANS...")
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    results = {}
    if args.eval_mnli_dev:
        mnli_result = evaluate_mnli_dev(model, tokenizer, device=device, batch_size=args.eval_batch_size)
        results["mnli_dev"] = {
            "accuracy": mnli_result["accuracy"],
            "correct": mnli_result["correct"],
            "total": mnli_result["total"],
        }
    
    if args.eval_hans:
        hans_result = evaluate_hans(model, tokenizer, device=device, batch_size=args.eval_batch_size)
        if hans_result:
            results["hans"] = {
                "accuracy": hans_result["accuracy"],
                "entailment_accuracy": hans_result["entailment_accuracy"],
                "non_entailment_accuracy": hans_result["non_entailment_accuracy"],
            }
    
    # Save results
    results_path = os.path.join(output_dir_phase3, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved evaluation results to: {results_path}")
    
    return output_dir_phase3, results


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate 3 scenarios")
    
    # Scenario selection
    parser.add_argument("--scenario", type=int, required=True, choices=[1, 2, 3],
                        help="Scenario number: 1=Full MNLI, 2=True-easy→Full, 3=True-easy→Easy→Hard")
    
    # Model args
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="Base model name (default: Qwen/Qwen2.5-0.5B, GLUE standard)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints")
    
    # JSON files (for scenarios 2 and 3)
    parser.add_argument("--true_easy_json", type=str, default=None,
                        help="Path to true-easy examples JSON (required for scenarios 2 and 3)")
    parser.add_argument("--stage1_easy_json", type=str, default=None,
                        help="Path to stage1 easy examples JSON (required for scenario 3)")
    
    # Training args (common) - Qwen2.5 standard hyperparameters
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of epochs for full training")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per device (Qwen2.5 standard: 8, with gradient_accumulation_steps=2, effective batch size=16)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate (Qwen2.5 fine-tuning standard: 2e-5)")
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--tokenize_batch_size", type=int, default=1000)
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["steps", "epoch"])
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=None)
    
    # Training args (scenario 3 specific)
    parser.add_argument("--num_epochs_easy", type=int, default=1,
                        help="Number of epochs for easy set training (scenario 3)")
    parser.add_argument("--num_epochs_hard", type=int, default=1,
                        help="Number of epochs for hard set training (scenario 3)")
    parser.add_argument("--lr_easy", type=float, default=2e-5,
                        help="Learning rate for easy set training (Qwen2.5 standard: 2e-5)")
    parser.add_argument("--lr_hard", type=float, default=2e-5,
                        help="Learning rate for hard set training (Qwen2.5 standard: 2e-5)")
    
    # Evaluation args
    parser.add_argument("--eval_mnli_dev", action="store_true", default=True,
                        help="Evaluate on MNLI dev")
    parser.add_argument("--eval_hans", action="store_true", default=True,
                        help="Evaluate on HANS")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use")
    
    args = parser.parse_args()
    
    # Validation
    if args.scenario in [2, 3] and args.true_easy_json is None:
        parser.error(f"--true_easy_json is required for scenario {args.scenario}")
    
    if args.scenario == 3 and args.stage1_easy_json is None:
        parser.error("--stage1_easy_json is required for scenario 3")
    
    # Run scenario
    if args.scenario == 1:
        output_dir, results = scenario1_full_mnli(args)
    elif args.scenario == 2:
        output_dir, results = scenario2_true_easy_full(args)
    elif args.scenario == 3:
        output_dir, results = scenario3_curriculum_3stage(args)
    
    # Final summary
    print("\n" + "="*70)
    print("Final Summary")
    print("="*70)
    print(f"Scenario: {args.scenario}")
    print(f"Final model: {output_dir}")
    print(f"\nEvaluation Results:")
    for dataset, metrics in results.items():
        print(f"  {dataset}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")
    print("="*70)


if __name__ == "__main__":
    main()

