"""
스크립트 3: Hypothesis-only 편향 태깅

Premise 없이 hypothesis만 사용하는 프롬프트로 평가하여,
hypothesis-only 상태에서도 정답을 맞추는 샘플을 bias 태그합니다.

Usage:
    python preliminary/script3_hypothesis_bias.py \
        --checkpoint_path ./checkpoints/pythia-160m-mnli-3epoch \
        --output_file ./results/hypothesis_bias_tags.jsonl \
        --batch_size 32
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm

# Project root 계산 (preliminary/script3_hypothesis_bias.py -> true_diff/)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.train import get_model_confidence_batch
from data.dataloader import load_mnli_raw, LABEL2TEXT


def make_hypothesis_only_prompt(hypothesis: str) -> str:
    """Premise 없이 hypothesis만 사용하는 프롬프트 생성"""
    return (
        "Decide the relationship for the following Hypothesis.\n"
        f"Hypothesis: {hypothesis}\n"
        "Answer:"
    )


def get_model_confidence_batch_hypothesis_only(model, tokenizer, hypotheses, device="cuda"):
    """
    Hypothesis-only 프롬프트로 배치 단위 예측 확률 계산
    """
    # Hypothesis-only 프롬프트 생성 (premise 없음)
    prompts = [make_hypothesis_only_prompt(h) for h in hypotheses]
    
    # 배치 토크나이즈
    prompt_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**prompt_inputs)
        batch_size = outputs.logits.shape[0]
        
        results = []
        for i in range(batch_size):
            last_token_logits = outputs.logits[i, -1, :]
            
            # 각 라벨의 첫 토큰 확률 계산
            label_probs = {}
            for label_text in LABEL2TEXT.values():
                # 공백 포함/미포함 모두 시도
                label_tokens_with_space = tokenizer.encode(f" {label_text}", add_special_tokens=False)
                label_tokens_without_space = tokenizer.encode(label_text, add_special_tokens=False)
                
                probs = []
                if len(label_tokens_with_space) > 0:
                    token_id = label_tokens_with_space[0]
                    prob = torch.softmax(last_token_logits, dim=-1)[token_id].item()
                    label_probs[f" {label_text}"] = prob
                
                if len(label_tokens_without_space) > 0:
                    token_id = label_tokens_without_space[0]
                    prob = torch.softmax(last_token_logits, dim=-1)[token_id].item()
                    if label_text not in label_probs or prob > label_probs.get(f" {label_text}", 0):
                        label_probs[label_text] = prob
            
            # 확률 정규화
            total_prob = sum(label_probs.values())
            if total_prob > 0:
                label_probs = {k: v / total_prob for k, v in label_probs.items()}
            
            # 가장 높은 확률의 라벨
            if label_probs:
                predicted_label = max(label_probs, key=label_probs.get)
                confidence = label_probs[predicted_label]
            else:
                predicted_label = "neutral"
                confidence = 0.0
            
            results.append((predicted_label, confidence, label_probs))
    
    return results


def evaluate_hypothesis_only(model, tokenizer, dataset, device="cuda", batch_size=32):
    """
    Hypothesis-only 프롬프트로 평가하여 정답 여부를 판단합니다.
    
    Returns:
        results: list[dict] with example_id, correct, predicted_label, true_label
    """
    model.to(device)
    model.eval()
    
    results = []
    total = len(dataset)
    
    print(f"Evaluating {total} examples with hypothesis-only prompt (batch_size={batch_size})...")
    
    for batch_start in tqdm(range(0, total, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, total)
        batch_data = dataset.select(range(batch_start, batch_end))
        
        hypotheses = []
        true_labels = []
        example_ids = []
        
        for idx, example in enumerate(batch_data):
            hypothesis = example["hypothesis"]
            true_label_idx = example.get("label", None)
            
            if true_label_idx is None or true_label_idx == -1:
                continue
            
            example_id = example.get("example_id", batch_start + idx)
            
            if isinstance(true_label_idx, int):
                true_label_text = LABEL2TEXT[int(true_label_idx)]
            else:
                true_label_text = str(true_label_idx).strip()
            
            hypotheses.append(hypothesis)
            true_labels.append(true_label_text.strip())
            example_ids.append(example_id)
        
        if len(hypotheses) == 0:
            continue
        
        # Get predictions with hypothesis-only prompt
        batch_results = get_model_confidence_batch_hypothesis_only(
            model, tokenizer, hypotheses, device
        )
        
        # Check correctness
        for (predicted_label, confidence, label_probs), true_label_text, example_id in zip(
            batch_results, true_labels, example_ids
        ):
            is_correct = predicted_label.strip() == true_label_text.strip()
            
            results.append({
                "example_id": example_id,
                "true_label": true_label_text,
                "predicted_label": predicted_label.strip(),
                "correct": is_correct,
                "hypothesis_only_bias": 1 if is_correct else 0,  # Bias tag: 1 if correct with hypothesis-only
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Hypothesis-only bias tagging")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to trained checkpoint (Pythia-160m trained for 3 epochs)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSONL file for bias tags")
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
    
    print("="*70)
    print("Hypothesis-only Bias Tagging")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Output file: {args.output_file}")
    print(f"Device: {device}")
    
    # Load model
    print(f"\n[1/3] Loading model from {args.checkpoint_path}...")
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.to(device)
    model.eval()
    print("  ✓ Model loaded")
    
    # Load MNLI train data
    print(f"\n[2/3] Loading MNLI training data...")
    train_dataset = load_mnli_raw(split="train", limit=None)
    train_dataset = train_dataset.map(
        lambda x, idx: {"example_id": idx},
        with_indices=True
    )
    print(f"  ✓ Loaded {len(train_dataset)} examples")
    
    # Evaluate with hypothesis-only prompt
    print(f"\n[3/3] Evaluating with hypothesis-only prompt...")
    results = evaluate_hypothesis_only(
        model=model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        device=device,
        batch_size=args.batch_size,
    )
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)
    print(f"\nSaving bias tags to {args.output_file}...")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✓ Saved {len(results)} bias tags")
    
    # Calculate statistics
    bias_count = sum(1 for r in results if r["hypothesis_only_bias"] == 1)
    bias_rate = bias_count / len(results) if results else 0.0
    
    print(f"\nHypothesis-only Bias Statistics:")
    print(f"  Total examples: {len(results)}")
    print(f"  Biased examples: {bias_count}")
    print(f"  Bias rate: {bias_rate:.4f} ({bias_rate*100:.2f}%)")
    
    print("\n" + "="*70)
    print("Hypothesis-only bias tagging completed!")
    print("="*70)


if __name__ == "__main__":
    main()

