"""
Model Evaluation Script

MNLI dev와 HANS 데이터셋에서 모델을 평가합니다.
여러 시나리오로 훈련된 모델들을 평가할 수 있습니다.

Usage:
    # Scenario 1: Full MNLI training
    python src/evaluate.py \
        --model_path ./checkpoints/full_mnli \
        --eval_mnli_dev \
        --eval_hans

    # Scenario 2: True-easy -> Full curriculum
    python src/evaluate.py \
        --model_path ./checkpoints/true_easy_full \
        --eval_mnli_dev \
        --eval_hans

    # Scenario 3: True-easy -> Easy -> Hard curriculum
    python src/evaluate.py \
        --model_path ./checkpoints/curriculum_3stage \
        --eval_mnli_dev \
        --eval_hans
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 상위 폴더를 path에 추가 (프로젝트 루트)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer
from data.dataloader import LABEL2TEXT, TEXT2LABEL, load_mnli_raw
from train import evaluate_examples, get_model_confidence_batch


def evaluate_mnli_dev(model, tokenizer, device="cuda", batch_size=32, max_examples=None):
    """
    MNLI validation_matched 데이터셋에서 평가합니다.
    
    Returns:
        dict: 평가 결과 (accuracy, predictions, etc.)
    """
    print("\n" + "="*70)
    print("Evaluating on MNLI validation_matched...")
    print("="*70)
    
    # MNLI dev 데이터 로드
    dataset = load_mnli_raw(split="validation_matched", limit=max_examples)
    print(f"  Dataset size: {len(dataset)}")
    
    # 평가 실행
    results = evaluate_examples(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        batch_size=batch_size,
        max_examples=max_examples,
    )
    
    # 정확도 계산
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / len(results) if results else 0.0
    
    # 라벨별 통계
    label_stats = {}
    for result in results:
        true_label = result["true_label"]
        if true_label not in label_stats:
            label_stats[true_label] = {"correct": 0, "total": 0}
        label_stats[true_label]["total"] += 1
        if result["correct"]:
            label_stats[true_label]["correct"] += 1
    
    # 결과 출력
    print(f"\n  Overall Accuracy: {accuracy:.4f} ({correct}/{len(results)})")
    print(f"\n  Label-wise Accuracy:")
    for label in sorted(label_stats.keys()):
        stats = label_stats[label]
        label_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"    {label}: {label_acc:.4f} ({stats['correct']}/{stats['total']})")
    
    # Classification report
    true_labels = [r["true_label"] for r in results]
    predicted_labels = [r["predicted_label"] for r in results]
    print(f"\n  Classification Report:")
    print(classification_report(true_labels, predicted_labels, labels=list(LABEL2TEXT.values()), zero_division=0))
    
    return {
        "dataset": "mnli_validation_matched",
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results),
        "label_stats": label_stats,
        "results": results,
    }


def evaluate_hans(model, tokenizer, device="cuda", batch_size=32, max_examples=None):
    """
    HANS 데이터셋에서 평가합니다.
    
    Returns:
        dict: 평가 결과
    """
    print("\n" + "="*70)
    print("Evaluating on HANS...")
    print("="*70)
    
    try:
        from datasets import load_dataset
        
        # HANS 데이터 로드
        print("  Loading HANS dataset...")
        hans_dataset = load_dataset("hans", split="validation" if max_examples is None else f"validation[:{max_examples}]")
        print(f"  Dataset size: {len(hans_dataset)}")
        
        # HANS는 "entailment" / "non-entailment" 라벨을 사용
        # MNLI의 "entailment", "neutral", "contradiction"과 매핑 필요
        # HANS의 경우 "entailment"만 entailment로, 나머지는 non-entailment
        # 하지만 모델은 3-way classification을 하므로, non-entailment를 neutral이나 contradiction으로 매핑
        
        # HANS 데이터를 MNLI 형식으로 변환
        results = []
        total = len(hans_dataset)
        
        print(f"  Evaluating {total} examples...")
        for batch_start in tqdm(range(0, total, batch_size), desc="Processing HANS batches"):
            batch_end = min(batch_start + batch_size, total)
            batch_data = hans_dataset.select(range(batch_start, batch_end))
            
            premises = []
            hypotheses = []
            true_labels_hans = []
            valid_indices = []
            
            for idx, example in enumerate(batch_data):
                premise = example["premise"]
                hypothesis = example["hypothesis"]
                true_label_hans = example["label"]  # "entailment" or "non-entailment"
                
                premises.append(premise)
                hypotheses.append(hypothesis)
                true_labels_hans.append(true_label_hans)
                valid_indices.append(batch_start + idx)
            
            if len(premises) == 0:
                continue
            
            # 배치로 예측
            batch_results = get_model_confidence_batch(
                model, tokenizer, premises, hypotheses, device
            )
            
            # 결과 처리
            for (predicted_label, confidence, label_probs), true_label_hans, orig_idx in zip(
                batch_results, true_labels_hans, valid_indices
            ):
                # HANS의 true_label을 MNLI 형식으로 변환
                # "entailment" -> "entailment"
                # "non-entailment" -> "neutral" (또는 contradiction, 여기서는 neutral로 처리)
                if true_label_hans == "entailment":
                    true_label_mnli = "entailment"
                else:  # non-entailment
                    true_label_mnli = "neutral"  # 또는 가장 높은 확률의 non-entailment 라벨 사용
                
                # 예측이 entailment면 entailment로, 아니면 non-entailment로 간주
                predicted_is_entailment = predicted_label.strip() == "entailment"
                true_is_entailment = true_label_hans == "entailment"
                is_correct = predicted_is_entailment == true_is_entailment
                
                # true_prob 계산 (entailment 확률)
                true_prob = label_probs.get("entailment", 0.0)
                
                results.append({
                    "example_id": orig_idx,
                    "true_label_hans": true_label_hans,
                    "true_label_mnli": true_label_mnli,
                    "predicted_label": predicted_label.strip(),
                    "predicted_is_entailment": predicted_is_entailment,
                    "correct": is_correct,
                    "confidence": confidence,
                    "entailment_prob": true_prob,
                    "all_probs": label_probs,
                })
        
        # 정확도 계산
        correct = sum(1 for r in results if r["correct"])
        accuracy = correct / len(results) if results else 0.0
        
        # 라벨별 통계
        entailment_correct = sum(1 for r in results if r["true_label_hans"] == "entailment" and r["correct"])
        entailment_total = sum(1 for r in results if r["true_label_hans"] == "entailment")
        non_entailment_correct = sum(1 for r in results if r["true_label_hans"] == "non-entailment" and r["correct"])
        non_entailment_total = sum(1 for r in results if r["true_label_hans"] == "non-entailment")
        
        entailment_acc = entailment_correct / entailment_total if entailment_total > 0 else 0.0
        non_entailment_acc = non_entailment_correct / non_entailment_total if non_entailment_total > 0 else 0.0
        
        print(f"\n  Overall Accuracy: {accuracy:.4f} ({correct}/{len(results)})")
        print(f"  Entailment Accuracy: {entailment_acc:.4f} ({entailment_correct}/{entailment_total})")
        print(f"  Non-entailment Accuracy: {non_entailment_acc:.4f} ({non_entailment_correct}/{non_entailment_total})")
        
        return {
            "dataset": "hans",
            "accuracy": accuracy,
            "correct": correct,
            "total": len(results),
            "entailment_accuracy": entailment_acc,
            "entailment_correct": entailment_correct,
            "entailment_total": entailment_total,
            "non_entailment_accuracy": non_entailment_acc,
            "non_entailment_correct": non_entailment_correct,
            "non_entailment_total": non_entailment_total,
            "results": results,
        }
        
    except Exception as e:
        print(f"  ERROR: Failed to load/evaluate HANS dataset: {e}")
        print(f"  Make sure 'datasets' library is installed and HANS dataset is available")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on MNLI dev and HANS")
    
    # Model args
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: cuda if available, else cpu)")
    
    # Evaluation args
    parser.add_argument("--eval_mnli_dev", action="store_true",
                        help="Evaluate on MNLI validation_matched")
    parser.add_argument("--eval_hans", action="store_true",
                        help="Evaluate on HANS")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to evaluate (for debugging)")
    
    # Output args
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save evaluation results (default: same as model_path)")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save detailed predictions to JSON file")
    
    args = parser.parse_args()
    
    # Device 설정
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Output directory
    if args.output_dir is None:
        args.output_dir = args.model_path
    
    print("="*70)
    print("Model Evaluation")
    print("="*70)
    print(f"Model path: {args.model_path}")
    print(f"Device: {device}")
    print(f"Output directory: {args.output_dir}")
    
    # 모델 로드
    print(f"\nLoading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.to(device)
    model.eval()
    print("  ✓ Model loaded successfully")
    
    # 평가 결과 저장
    all_results = {
        "model_path": args.model_path,
        "evaluations": {},
    }
    
    # MNLI dev 평가
    if args.eval_mnli_dev:
        mnli_results = evaluate_mnli_dev(
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.batch_size,
            max_examples=args.max_examples,
        )
        all_results["evaluations"]["mnli_dev"] = {
            "accuracy": mnli_results["accuracy"],
            "correct": mnli_results["correct"],
            "total": mnli_results["total"],
            "label_stats": mnli_results["label_stats"],
        }
        
        if args.save_predictions:
            predictions_path = os.path.join(args.output_dir, "mnli_dev_predictions.json")
            with open(predictions_path, "w", encoding="utf-8") as f:
                json.dump(mnli_results["results"], f, indent=2, ensure_ascii=False)
            print(f"  ✓ Saved predictions to: {predictions_path}")
    
    # HANS 평가
    if args.eval_hans:
        hans_results = evaluate_hans(
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.batch_size,
            max_examples=args.max_examples,
        )
        if hans_results:
            all_results["evaluations"]["hans"] = {
                "accuracy": hans_results["accuracy"],
                "correct": hans_results["correct"],
                "total": hans_results["total"],
                "entailment_accuracy": hans_results["entailment_accuracy"],
                "non_entailment_accuracy": hans_results["non_entailment_accuracy"],
            }
            
            if args.save_predictions:
                predictions_path = os.path.join(args.output_dir, "hans_predictions.json")
                with open(predictions_path, "w", encoding="utf-8") as f:
                    json.dump(hans_results["results"], f, indent=2, ensure_ascii=False)
                print(f"  ✓ Saved predictions to: {predictions_path}")
    
    # 결과 저장
    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved evaluation summary to: {summary_path}")
    
    # 요약 출력
    print("\n" + "="*70)
    print("Evaluation Summary")
    print("="*70)
    for dataset_name, metrics in all_results["evaluations"].items():
        print(f"\n{dataset_name}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

