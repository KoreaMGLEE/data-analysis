"""
MNLI 데이터로 Pythia 30M 모델을 훈련하고, 
높은 확신으로 정답을 맞추는 쉬운 예제를 식별하는 코드
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import numpy as np
from tqdm import tqdm

# 상위 폴더의 data 모듈 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataloader import (
    tokenize_fn,
    collate_fn,
    LABEL2TEXT,
    TEXT2LABEL,
    prepare_dataset,
    load_mnli_raw,
)


def train_model(
    model_name="EleutherAI/pythia-30m",
    output_dir="../checkpoints/pythia-30m-mnli",
    num_train_epochs=1,
    batch_size=16,
    learning_rate=5e-5,
    max_length=256,
    train_limit=None,  # 디버깅용: 훈련 데이터 제한
    save_steps=500,
    eval_steps=500,
    eval_strategy="epoch",  # "steps" 또는 "epoch"
    num_proc=None,  # 토크나이징 멀티프로세싱 프로세스 수 (None이면 CPU 코어 수 사용)
    tokenize_batch_size=1000,  # 배치 토크나이징에 사용할 배치 크기
    save_total_limit=None,  # 최대 저장할 체크포인트 수 (None = 모두 저장)
):
    """Pythia 30M 모델을 MNLI 데이터로 훈련합니다."""
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # pad token 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
    )
    
    print("Loading MNLI training data...")
    train_dataset = prepare_dataset(
        tokenizer, 
        split="train",
        max_length=max_length,
        limit=train_limit,
        num_proc=num_proc,
        batch_size=tokenize_batch_size,
    )
    
    # 쉬운 예제 찾기용 원본 training 데이터 로드
    train_dataset_raw = load_mnli_raw(split="train", limit=train_limit)
    
    # Validation 데이터도 로드 (평가용 - 토크나이즈된 버전)
    print("Loading MNLI validation data...")
    val_dataset_tokenized = prepare_dataset(
        tokenizer,
        split="validation_matched",
        max_length=max_length,
        limit=1000,  # validation은 작은 샘플만 사용
        num_proc=num_proc,
        batch_size=tokenize_batch_size,
    )
    
    # Training arguments
    # load_best_model_at_end=True를 사용하려면 save_strategy와 eval_strategy가 일치해야 함
    save_strategy = eval_strategy  # eval_strategy와 동일하게 설정
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=1000,
        save_steps=save_steps if save_strategy == "steps" else None,
        save_strategy=save_strategy,  # eval_strategy와 일치
        eval_steps=eval_steps if eval_strategy == "steps" else None,
        eval_strategy=eval_strategy,  # "steps" 또는 "epoch"
        save_total_limit=save_total_limit,  # None이면 모든 체크포인트 저장
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=100,
        fp16=False,  # 30M 모델은 fp32로도 충분히 빠름
        report_to=None,  # wandb 등 비활성화
    )
    
    # Custom data collator
    def data_collator(batch):
        return collate_fn(batch, tokenizer)
    
    # Trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset_tokenized,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer, trainer, train_dataset_raw


def get_model_confidence_batch(model, tokenizer, premises, hypotheses, device="cuda"):
    """
    배치 단위로 모델의 예측 확률과 확신을 계산합니다 (훨씬 빠름).
    
    Args:
        model: 모델
        tokenizer: 토크나이저
        premises: 프리미스 리스트
        hypotheses: 가설 리스트
        device: 디바이스
    
    Returns:
        results: 리스트 [(predicted_label, confidence, label_probs), ...]
    """
    from data.dataloader import make_prompt
    
    # 모든 프롬프트 생성
    prompts = [make_prompt(p, h) for p, h in zip(premises, hypotheses)]
    
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
        # 배치로 forward pass
        outputs = model(**prompt_inputs)
        # 각 예제의 마지막 토큰 위치의 logits
        batch_size = outputs.logits.shape[0]
        next_token_logits = outputs.logits[range(batch_size), -1, :]  # [batch_size, vocab_size]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)  # [batch_size, vocab_size]
    
    # 각 라벨의 첫 토큰 ID 준비
    label_first_token_ids = {}
    for label_text in LABEL2TEXT.values():
        label_clean = label_text.strip()
        label_token_ids = tokenizer.encode(label_clean, add_special_tokens=False)
        if len(label_token_ids) > 0:
            label_first_token_ids[label_text] = label_token_ids[0]
    
    # 각 예제에 대해 결과 계산
    results = []
    for i in range(batch_size):
        probs = next_token_probs[i]
        
        # 각 라벨의 확률 계산
        label_probs = {}
        for label_text, token_id in label_first_token_ids.items():
            label_probs[label_text] = probs[token_id].item()
        
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


def find_easy_examples(
    model,
    tokenizer,
    dataset,
    confidence_threshold=0.7,
    device="cuda",
    max_examples=None,
    batch_size=32,  # 배치 크기
):
    """
    모델이 높은 확신으로 정답을 맞추는 쉬운 예제를 찾습니다 (배치 처리로 빠름).
    
    Args:
        model: 훈련된 모델
        tokenizer: 토크나이저
        dataset: 평가할 데이터셋 (HuggingFace Dataset)
        confidence_threshold: 최소 확신 임계값 (기본 0.8)
        device: 사용할 디바이스
        max_examples: 최대 평가할 예제 수 (None이면 전체)
        batch_size: 배치 크기 (GPU 메모리에 따라 조정)
    
    Returns:
        easy_examples: 쉬운 예제 리스트 (dict 형태)
    """
    model.to(device)
    model.eval()
    
    easy_examples = []
    
    # max_examples가 지정된 경우 데이터셋 제한
    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    
    total = len(dataset)
    print(f"Evaluating {total} examples to find easy examples (batch_size={batch_size})...")
    
    # 배치 단위로 처리
    for batch_start in tqdm(range(0, total, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, total)
        batch_indices = list(range(batch_start, batch_end))
        batch_data = dataset.select(batch_indices)
        
        # 배치 데이터 준비
        premises = []
        hypotheses = []
        true_labels = []
        valid_indices = []
        
        for idx, example in enumerate(batch_data):
            premise = example["premise"]
            hypothesis = example["hypothesis"]
            true_label_idx = example["label"]
            
            if true_label_idx == -1:
                continue
            
            premises.append(premise)
            hypotheses.append(hypothesis)
            true_labels.append(LABEL2TEXT[int(true_label_idx)])
            valid_indices.append((batch_start + idx, example))
        
        if len(premises) == 0:
            continue
        
        # 배치로 예측
        results = get_model_confidence_batch(
            model, tokenizer, premises, hypotheses, device
        )
        
        # 결과 처리
        for (predicted_label, confidence, label_probs), true_label_text, (orig_idx, example) in zip(
            results, true_labels, valid_indices
        ):
            # 정답을 맞추고 confidence가 임계값 이상인 경우
            if predicted_label == true_label_text and confidence >= confidence_threshold:
                easy_examples.append({
                    "premise": example["premise"],
                    "hypothesis": example["hypothesis"],
                    "true_label": true_label_text,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "all_probs": label_probs,
                    "example_id": orig_idx,
                })
    
    print(f"\nFound {len(easy_examples)} easy examples (confidence >= {confidence_threshold})")
    return easy_examples


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory. None이면 자동 생성: checkpoints/{model}-lr{lr}-ep{epochs}")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--train_limit", type=int, default=None, help="Limit training examples for debugging")
    parser.add_argument("--confidence_threshold", type=float, default=0.8)
    parser.add_argument("--eval_limit", type=int, default=None, help="Number of examples to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_training", action="store_true", help="Skip training if model already exists")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to existing checkpoint")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes for tokenization (None = use all CPU cores)")
    parser.add_argument("--tokenize_batch_size", type=int, default=1000, help="Batch size for tokenization")
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["steps", "epoch"], help="Evaluation strategy: 'steps' or 'epoch'")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps (only used when eval_strategy='steps')")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--save_total_limit", type=int, default=None, help="Maximum number of checkpoints to save (None = save all)")
    
    args = parser.parse_args()
    
    # output_dir가 지정되지 않았으면 자동으로 생성 (실험 설정 포함)
    if args.output_dir is None:
        model_short = args.model_name.split("/")[-1] if "/" in args.model_name else args.model_name
        args.output_dir = f"./checkpoints/{model_short}-lr{args.learning_rate}-ep{args.num_epochs}"
    
    print(f"Using device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    
    # 1. 모델 훈련 (또는 기존 모델 로드)
    if args.skip_training and args.checkpoint_path:
        print(f"Loading model from {args.checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    else:
        model, tokenizer, trainer, train_dataset_raw = train_model(
            model_name=args.model_name,
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            train_limit=args.train_limit,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            eval_strategy=args.eval_strategy,
            num_proc=args.num_proc,
            tokenize_batch_size=args.tokenize_batch_size,
            save_total_limit=args.save_total_limit,
        )
    
    # 2. 쉬운 예제 찾기
    print("\n" + "="*50)
    print("Finding easy examples in training data...")
    print("="*50)
    
    # skip_training인 경우 train_dataset_raw가 없으므로 로드
    if args.skip_training and args.checkpoint_path:
        train_dataset_raw = load_mnli_raw(split="train", limit=args.eval_limit)
    
    # 모델을 device로 이동
    model.to(args.device)
    
    easy_examples = find_easy_examples(
        model=model,
        tokenizer=tokenizer,
        dataset=train_dataset_raw,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        max_examples=args.eval_limit,
        batch_size=32,  # GPU 메모리에 따라 조정 가능
    )
    
    # 3. 결과 저장
    import json
    output_file = f"./easy_examples_confidence_{args.confidence_threshold}_{args.num_epochs}_{args.learning_rate}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(easy_examples, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(easy_examples)} easy examples to {output_file}")
    
    # 4. 샘플 출력
    if easy_examples:
        print("\n" + "="*50)
        print("Sample easy examples:")
        print("="*50)
        for i, ex in enumerate(easy_examples[:5]):
            print(f"\nExample {i+1}:")
            print(f"Premise: {ex['premise']}")
            print(f"Hypothesis: {ex['hypothesis']}")
            print(f"True label: {ex['true_label']}")
            print(f"Predicted label: {ex['predicted_label']}")
            print(f"Confidence: {ex['confidence']:.4f}")


if __name__ == "__main__":
    main()

