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
    
    # Validation 데이터도 로드 (평가용)
    print("Loading MNLI validation data...")
    val_dataset = prepare_dataset(
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
        save_total_limit=2,
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
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer, trainer


def get_model_confidence(model, tokenizer, premise, hypothesis, device="cuda"):
    """모델의 예측 확률과 확신을 계산합니다."""
    from data.dataloader import make_prompt
    
    prompt = make_prompt(premise, hypothesis)
    
    # prompt 토크나이즈
    prompt_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        # prompt에 대한 다음 토큰 예측
        outputs = model(**prompt_inputs)
        next_token_logits = outputs.logits[0, -1, :]  # 마지막 위치의 logits
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
    
    # 각 라벨 텍스트의 첫 토큰 확률 계산
    label_probs = {}
    for label_text in LABEL2TEXT.values():
        # 라벨 텍스트 토크나이즈 (공백 제거 후)
        label_clean = label_text.strip()
        label_token_ids = tokenizer.encode(label_clean, add_special_tokens=False)
        
        if len(label_token_ids) > 0:
            # 첫 토큰의 확률을 사용
            first_token_id = label_token_ids[0]
            prob = next_token_probs[first_token_id].item()
            label_probs[label_text] = prob
    
    # 확률 정규화 (선택사항 - 합이 1이 되도록)
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
    
    return predicted_label, confidence, label_probs


def find_easy_examples(
    model,
    tokenizer,
    dataset,
    confidence_threshold=0.7,
    device="cuda",
    max_examples=None,
):
    """
    모델이 높은 확신으로 정답을 맞추는 쉬운 예제를 찾습니다.
    
    Args:
        model: 훈련된 모델
        tokenizer: 토크나이저
        dataset: 평가할 데이터셋 (HuggingFace Dataset)
        confidence_threshold: 최소 확신 임계값 (기본 0.8)
        device: 사용할 디바이스
        max_examples: 최대 평가할 예제 수 (None이면 전체)
    
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
    print(f"Evaluating {total} examples to find easy examples...")
    
    for i, example in enumerate(tqdm(dataset, total=total)):
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        true_label_idx = example["label"]
        
        if true_label_idx == -1:
            continue
        
        true_label_text = LABEL2TEXT[int(true_label_idx)]
        
        # 모델 예측
        predicted_label, confidence, label_probs = get_model_confidence(
            model, tokenizer, premise, hypothesis, device
        )
        
        # 정답을 맞추고 confidence가 임계값 이상인 경우
        if predicted_label == true_label_text and confidence >= confidence_threshold:
            easy_examples.append({
                "premise": premise,
                "hypothesis": hypothesis,
                "true_label": true_label_text,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "all_probs": label_probs,
                "example_id": i,
            })
    
    print(f"\nFound {len(easy_examples)} easy examples (confidence >= {confidence_threshold})")
    return easy_examples


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/pythia-70m-mnli")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--train_limit", type=int, default=None, help="Limit training examples for debugging")
    parser.add_argument("--confidence_threshold", type=float, default=0.8)
    parser.add_argument("--eval_limit", type=int, default=1000, help="Number of examples to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_training", action="store_true", help="Skip training if model already exists")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to existing checkpoint")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes for tokenization (None = use all CPU cores)")
    parser.add_argument("--tokenize_batch_size", type=int, default=1000, help="Batch size for tokenization")
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["steps", "epoch"], help="Evaluation strategy: 'steps' or 'epoch'")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps (only used when eval_strategy='steps')")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # 1. 모델 훈련 (또는 기존 모델 로드)
    if args.skip_training and args.checkpoint_path:
        print(f"Loading model from {args.checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    else:
        model, tokenizer, trainer = train_model(
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
        )
    
    # 2. 쉬운 예제 찾기
    print("\n" + "="*50)
    print("Finding easy examples...")
    print("="*50)
    
    # Validation 데이터셋 로드 (원본 데이터, 토크나이즈 전)
    val_dataset = load_mnli_raw(split="validation_matched", limit=args.eval_limit)
    
    # 모델을 device로 이동
    model.to(args.device)
    
    easy_examples = find_easy_examples(
        model=model,
        tokenizer=tokenizer,
        dataset=val_dataset,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        max_examples=args.eval_limit,
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

