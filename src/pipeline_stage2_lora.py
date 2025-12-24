"""
Stage 2: Conflicting Proxy Training Pipeline with LoRA

이 스크립트는 stage1에서 찾은 easy examples를 제외하고 더 큰 모델을 LoRA로 학습한 후,
학습된 모델로 다시 easy example mining을 수행합니다.

How to run (with training):
    python src/pipeline_stage2_lora.py \
        --stage1_easy_json ./easy_examples_confidence_0.8_1_5e-05.json \
        --stage2_model_name EleutherAI/pythia-160m \
        --stage2_output_dir ./checkpoints/stage2_lora/pythia-160m \
        --confidence_threshold 0.8 \
        --num_epochs 1 \
        --batch_size 16 \
        --learning_rate 2e-4

How to run (skip training, only re-evaluate):
    python src/pipeline_stage2_lora.py \
        --stage1_easy_json ./easy_examples_confidence_0.8_1_5e-05.json \
        --stage2_output_dir ./checkpoints/stage2_lora/pythia-160m-lr2e-04-ep1 \
        --skip_training \
        --confidence_threshold 0.8

Smoke test:
    python src/pipeline_stage2_lora.py \
        --stage1_easy_json <path_to_easy_json> \
        --stage2_model_name EleutherAI/pythia-14m \
        --confidence_threshold 0.8 \
        --train_limit 2000 \
        --num_epochs 1
"""

import os
import sys
import json
import argparse
import torch
import tempfile

# 상위 폴더를 path에 추가 (프로젝트 루트)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 같은 폴더(src)에 있으므로 직접 import
from train import find_easy_examples
from data.dataloader import load_mnli_raw, prepare_dataset, tokenize_fn_batch, collate_fn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, PeftModel


def train_model_lora(
    model_name="EleutherAI/pythia-160m",
    output_dir="./checkpoints/stage2_lora/pythia-160m",
    num_train_epochs=1,
    batch_size=16,
    learning_rate=2e-4,  # LoRA는 일반적으로 더 높은 학습률 사용 (2e-4 ~ 5e-4)
    max_length=256,
    train_limit=None,
    save_steps=500,
    eval_steps=500,
    eval_strategy="epoch",
    num_proc=None,
    tokenize_batch_size=1000,
    save_total_limit=None,
    exclude_ids_json=None,
    exclude_id_field="example_id",
    # LoRA 하이퍼파라미터
    lora_r=8,  # LoRA rank (일반적으로 8, 16, 32)
    lora_alpha=16,  # LoRA alpha (보통 rank의 2배)
    lora_dropout=0.1,  # LoRA dropout
    lora_target_modules=None,  # None이면 자동 감지
):
    """
    Pythia 모델을 LoRA로 MNLI 데이터에 훈련합니다.
    
    LoRA 하이퍼파라미터는 Llama 같은 decoder 모델 튜닝에 사용되는 값들을 참고했습니다:
    - r: 8 (rank, 일반적인 기본값)
    - alpha: 16 (보통 rank의 2배)
    - dropout: 0.1
    - learning_rate: 2e-4 (LoRA는 전체 파인튜닝보다 높은 학습률 사용)
    """
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # pad token 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 모델 로드 (float32로, LoRA 적용 후 필요시 fp16 가능)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
    )
    
    # LoRA target_modules 자동 감지 (Pythia/GPTNeoX 모델의 경우)
    if lora_target_modules is None:
        # Pythia/GPTNeoX 모델의 attention 모듈 이름 확인
        model_modules = [name for name, _ in model.named_modules()]
        # 일반적으로 query_key_value, dense 등이 있음
        if any("query_key_value" in name for name in model_modules):
            lora_target_modules = ["query_key_value", "dense"]
        elif any("q_proj" in name for name in model_modules):
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        else:
            # 기본값으로 attention 관련 모듈 찾기
            attention_modules = [name for name in model_modules if "attention" in name.lower() and ("query" in name.lower() or "dense" in name.lower())]
            if attention_modules:
                lora_target_modules = list(set([name.split(".")[-1] for name in attention_modules[:4]]))
            else:
                raise ValueError("Could not auto-detect LoRA target modules. Please specify --lora_target_modules manually.")
        
        print(f"Auto-detected LoRA target modules: {lora_target_modules}")
    
    # LoRA 설정 (Llama 등 decoder 모델 튜닝 방식 참고)
    lora_config = LoraConfig(
        r=lora_r,  # rank
        lora_alpha=lora_alpha,  # alpha (보통 rank의 2배)
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",  # bias는 학습하지 않음
        task_type="CAUSAL_LM",
    )
    
    # LoRA 적용
    model = get_peft_model(model, lora_config)
    
    # 학습 가능한 파라미터 수 출력
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable%: {100 * trainable_params / all_params:.4f}%")
    
    print("Loading MNLI training data...")
    # 먼저 raw 데이터 로드 (필터링을 위해)
    train_dataset_raw = load_mnli_raw(split="train", limit=train_limit)
    
    # exclude_ids가 있으면 필터링
    exclude_ids_set = None
    if exclude_ids_json:
        with open(exclude_ids_json, "r", encoding="utf-8") as f:
            exclude_data = json.load(f)
        
        # JSON이 리스트인 경우와 딕셔너리 리스트인 경우 처리
        if isinstance(exclude_data, list):
            if len(exclude_data) > 0 and isinstance(exclude_data[0], dict):
                # 딕셔너리 리스트: 특정 필드 추출
                exclude_ids_set = set(item.get(exclude_id_field) for item in exclude_data if exclude_id_field in item)
            else:
                # 단순 ID 리스트
                exclude_ids_set = set(exclude_data)
        else:
            raise ValueError(f"exclude_ids_json must be a list, got {type(exclude_data)}")
        
        original_size = len(train_dataset_raw)
        # row index 기반으로 필터링 (example_id가 row index로 저장된 경우)
        train_dataset_raw = train_dataset_raw.filter(lambda x, idx: idx not in exclude_ids_set, with_indices=True)
        filtered_size = len(train_dataset_raw)
        print(f"Excluded {original_size - filtered_size} examples (from {original_size} total)")
    
    # 필터링된/원본 raw 데이터를 토크나이즈
    if exclude_ids_set is not None:
        # 필터링된 raw 데이터를 직접 토크나이즈
        train_dataset = train_dataset_raw.map(
            lambda examples: tokenize_fn_batch(examples, tokenizer, max_length=max_length),
            batched=True,
            batch_size=tokenize_batch_size,
            remove_columns=train_dataset_raw.column_names,
            desc="Tokenizing filtered training data",
            num_proc=num_proc,
        )
        train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    else:
        # exclude가 없으면 기존 방식 사용
        from data.dataloader import prepare_dataset
        train_dataset = prepare_dataset(
            tokenizer, 
            split="train",
            max_length=max_length,
            limit=train_limit,
            num_proc=num_proc,
            batch_size=tokenize_batch_size,
        )
    
    # Validation 데이터도 로드
    print("Loading MNLI validation data...")
    from data.dataloader import prepare_dataset
    val_dataset_tokenized = prepare_dataset(
        tokenizer,
        split="validation_matched",
        max_length=max_length,
        limit=1000,
        num_proc=num_proc,
        batch_size=tokenize_batch_size,
    )
    
    # Training arguments (LoRA에 맞춘 하이퍼파라미터)
    save_strategy = eval_strategy
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,  # LoRA는 높은 학습률 사용 (2e-4 ~ 5e-4)
        logging_steps=1000,
        save_steps=save_steps if save_strategy == "steps" else None,
        save_strategy=save_strategy,
        eval_steps=eval_steps if eval_strategy == "steps" else None,
        eval_strategy=eval_strategy,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=100,
        fp16=False,  # 필요시 True로 변경 가능
        report_to=None,
        # Recommended optimizer hyperparameters (decoder 모델 튜닝 방식 참고)
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        weight_decay=0.1,
        # Recommended scheduler
        lr_scheduler_type="cosine",
        # Recommended gradient clipping
        max_grad_norm=1.0,
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
    
    print("Starting LoRA training...")
    trainer.train()
    
    print(f"Saving LoRA adapter to {output_dir}")
    # LoRA adapter만 저장 (base model은 저장하지 않음)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # base_model_name_or_path가 adapter_config.json에 저장되도록 확인
    # (PEFT가 자동으로 처리하지만, 명시적으로 확인)
    adapter_config_path = os.path.join(output_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
        if "base_model_name_or_path" not in adapter_config or adapter_config["base_model_name_or_path"] is None:
            adapter_config["base_model_name_or_path"] = model_name
            with open(adapter_config_path, "w") as f:
                json.dump(adapter_config, f, indent=2)
    
    return model, tokenizer, trainer, train_dataset_raw


def load_model_with_lora(base_model_name, lora_adapter_path, device="cuda"):
    """
    Base 모델에 LoRA adapter를 로드합니다.
    """
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float32,
    )
    
    print(f"Loading LoRA adapter from: {lora_adapter_path}")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    model = model.merge_and_unload()  # LoRA weights를 base model에 병합
    
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Train with LoRA (easy examples excluded), then mine again")
    
    # Stage 2 specific args
    parser.add_argument("--stage1_easy_json", type=str, required=True,
                        help="Path to stage1 easy examples JSON file")
    parser.add_argument("--stage2_model_name", type=str, default=None,
                        help="Base model name for stage2 LoRA training (larger model). Required if not using --skip_training")
    parser.add_argument("--stage2_output_dir", type=str, default=None,
                        help="Output directory for stage2 LoRA adapter")
    parser.add_argument("--confidence_threshold", type=float, default=0.8,
                        help="Confidence threshold for easy example mining (ignored if --no_confidence_check is set)")
    parser.add_argument("--no_confidence_check", action="store_true",
                        help="Only check if prediction is correct, ignore confidence threshold")
    
    # Training args
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate for LoRA (typically higher than full fine-tuning: 2e-4 ~ 5e-4)")
    parser.add_argument("--train_limit", type=int, default=None,
                        help="Limit training examples for debugging")
    parser.add_argument("--eval_limit", type=int, default=None,
                        help="Limit examples for easy mining")
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--tokenize_batch_size", type=int, default=1000)
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["steps", "epoch"])
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--exclude_id_field", type=str, default="example_id",
                        help="Field name in JSON for example IDs")
    
    # LoRA-specific args
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank (default: 8, typical values: 8, 16, 32)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha (default: 16, typically 2x rank)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout (default: 0.1)")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None,
                        help="LoRA target modules (default: auto-detect)")
    
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and load existing LoRA adapter from --stage2_output_dir")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: cuda if available, else cpu)")
    
    args = parser.parse_args()
    
    # Validation
    if not args.skip_training and args.stage2_model_name is None:
        parser.error("--stage2_model_name is required when not using --skip_training")
    
    print("="*70)
    print("Stage 2: Conflicting Proxy Training Pipeline with LoRA")
    print("="*70)
    
    # Step a) Load stage1 easy JSON and extract exclude_ids
    print(f"\n[Step a] Loading stage1 easy examples from: {args.stage1_easy_json}")
    with open(args.stage1_easy_json, "r", encoding="utf-8") as f:
        stage1_easy_data = json.load(f)
    
    if not isinstance(stage1_easy_data, list):
        raise ValueError(f"Expected list in JSON, got {type(stage1_easy_data)}")
    
    if len(stage1_easy_data) == 0:
        raise ValueError("Stage1 easy JSON is empty")
    
    # Extract exclude_ids
    if isinstance(stage1_easy_data[0], dict):
        exclude_ids = [item.get(args.exclude_id_field) for item in stage1_easy_data 
                      if args.exclude_id_field in item]
    else:
        exclude_ids = stage1_easy_data
    
    exclude_ids_set = set(exclude_ids)
    print(f"  Found {len(exclude_ids_set)} unique IDs to exclude")
    
    # Step b) Train model with LoRA (easy examples excluded) or load existing model
    temp_exclude_json = None
    if args.skip_training:
        # 기존 LoRA adapter 로드
        if args.stage2_output_dir is None:
            raise ValueError("--stage2_output_dir must be provided when using --skip_training")
        
        if args.stage2_model_name is None:
            # adapter_config.json에서 base model 이름 읽기
            adapter_config_path = os.path.join(args.stage2_output_dir, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                with open(adapter_config_path, "r") as f:
                    adapter_config = json.load(f)
                    args.stage2_model_name = adapter_config.get("base_model_name_or_path")
                    if args.stage2_model_name:
                        print(f"  Found base model name in adapter_config: {args.stage2_model_name}")
        
        if args.stage2_model_name is None:
            raise ValueError("--stage2_model_name must be provided when adapter_config.json doesn't contain base_model_name_or_path")
        
        print(f"\n[Step b] Loading existing LoRA adapter from: {args.stage2_output_dir}")
        print(f"  Base model: {args.stage2_model_name}")
        print(f"  Skipping training...")
        
        model, tokenizer = load_model_with_lora(args.stage2_model_name, args.stage2_output_dir)
        
        print(f"  ✓ Model loaded successfully")
        train_dataset_raw = None
    else:
        # LoRA로 모델 훈련
        temp_exclude_json = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
        json.dump([{"example_id": id} for id in exclude_ids_set], temp_exclude_json, indent=2)
        temp_exclude_json.close()
        
        print(f"\n[Step b] Training stage2 model with LoRA: {args.stage2_model_name}")
        print(f"  LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        print(f"  Excluding {len(exclude_ids_set)} easy examples from training")
        
        # Generate output_dir if not provided
        if args.stage2_output_dir is None:
            if args.stage2_model_name:
                model_short = args.stage2_model_name.split("/")[-1] if "/" in args.stage2_model_name else args.stage2_model_name
            else:
                model_short = "unknown_model"
            args.stage2_output_dir = f"./checkpoints/stage2_lora/{model_short}-lr{args.learning_rate}-ep{args.num_epochs}"
        
        print(f"  Output directory: {args.stage2_output_dir}")
        
        # Train with LoRA and exclusions
        try:
            model, tokenizer, trainer, train_dataset_raw = train_model_lora(
                model_name=args.stage2_model_name,
                output_dir=args.stage2_output_dir,
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
                exclude_ids_json=temp_exclude_json.name,
                exclude_id_field="example_id",
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                lora_target_modules=args.lora_target_modules,
            )
            
            print(f"\n  ✓ Stage2 LoRA training completed")
            if train_dataset_raw is not None:
                print(f"    Remaining training examples after exclusion: {len(train_dataset_raw)}")
            
        finally:
            # Clean up temp file
            if temp_exclude_json:
                os.unlink(temp_exclude_json.name)
    
    # Step c) Load Stage1 easy examples only
    print(f"\n[Step c] Loading Stage1 easy examples for re-evaluation...")
    full_train_raw = load_mnli_raw(split="train", limit=None)
    
    print(f"  Filtering Stage1 easy examples (IDs: {len(exclude_ids_set)} examples)...")
    stage1_easy_dataset = full_train_raw.filter(
        lambda x, idx: idx in exclude_ids_set,
        with_indices=True
    )
    print(f"  Loaded {len(stage1_easy_dataset)} Stage1 easy examples for re-evaluation")
    
    # Step d) Find real easy examples with stage2 model
    print(f"\n[Step d] Re-evaluating Stage1 easy examples with stage2 model...")
    print(f"  Stage2 model will re-evaluate {len(stage1_easy_dataset)} examples that stage1 model found easy")
    
    # Device 설정
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"  Using device: {device}")
    
    # Stage1 easy examples 중에서 Stage2 모델이 맞추는 것들만 찾기
    easy_examples_stage2 = find_easy_examples(
        model=model,
        tokenizer=tokenizer,
        dataset=stage1_easy_dataset,
        confidence_threshold=args.confidence_threshold,
        device=device,
        max_examples=None,
        batch_size=32,
        use_confidence=not args.no_confidence_check,
    )
    
    print(f"\n  ✓ Found {len(easy_examples_stage2)} real easy examples (stage2 model correctly predicted)")
    print(f"     Out of {len(stage1_easy_dataset)} Stage1 easy examples")
    if len(stage1_easy_dataset) > 0:
        print(f"     Success rate: {len(easy_examples_stage2)/len(stage1_easy_dataset)*100:.2f}%")
    
    # Step e) Save results
    print(f"\n[Step e] Saving real easy examples...")
    
    # 모델명 추출
    if args.stage2_model_name:
        model_short = args.stage2_model_name.split("/")[-1] if "/" in args.stage2_model_name else args.stage2_model_name
    elif args.stage2_output_dir:
        dir_name = os.path.basename(args.stage2_output_dir.rstrip("/"))
        if "-lr" in dir_name:
            model_short = dir_name.split("-lr")[0]
        else:
            model_short = dir_name
    else:
        model_short = "unknown_model"
    
    threshold_str = f"conf{args.confidence_threshold}" if not args.no_confidence_check else "correct_only"
    
    easy_short = args.stage1_easy_json.split('_0.8')[-1] if '_0.8' in args.stage1_easy_json else ""
    output_json = f"real_easy_examples_{threshold_str}_stage2_lora_{model_short}_{easy_short}.json"
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(easy_examples_stage2, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved to: {output_json}")
    
    # Summary
    print("\n" + "="*70)
    print("Stage 2 Pipeline Summary (LoRA)")
    print("="*70)
    print(f"Stage1 easy examples: {len(exclude_ids_set)}")
    if train_dataset_raw is not None:
        print(f"Training data (hard examples): {len(train_dataset_raw)}")
    else:
        print(f"Training data: (skipped - using existing model)")
    print(f"Stage1 easy examples re-evaluated: {len(stage1_easy_dataset)}")
    print(f"Real easy examples (stage2 model also correct): {len(easy_examples_stage2)}")
    if len(stage1_easy_dataset) > 0:
        print(f"Success rate: {len(easy_examples_stage2)/len(stage1_easy_dataset)*100:.2f}%")
    print(f"Output JSON: {output_json}")
    print("="*70)


if __name__ == "__main__":
    main()

