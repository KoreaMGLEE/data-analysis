"""
Stage 2: Conflicting Proxy Training Pipeline

이 스크립트는 stage1에서 찾은 easy examples를 제외하고 더 큰 모델로 학습한 후,
학습된 모델로 다시 easy example mining을 수행합니다.

How to run (with training):
    python src/pipeline_stage2_conflicting_train.py \
        --stage1_easy_json ./easy_examples_confidence_0.8_1_5e-05.json \
        --stage2_model_name EleutherAI/pythia-160m \
        --stage2_output_dir ./checkpoints/stage2/pythia-160m \
        --confidence_threshold 0.8 \
        --num_epochs 1 \
        --batch_size 16 \
        --learning_rate 5e-5

How to run (skip training, only re-evaluate):
    python src/pipeline_stage2_conflicting_train.py \
        --stage1_easy_json ./easy_examples_confidence_0.8_1_5e-05.json \
        --stage2_output_dir ./checkpoints/stage2/pythia-160m-lr5e-05-ep1 \
        --skip_training \
        --confidence_threshold 0.8

Smoke test:
    python src/pipeline_stage2_conflicting_train.py \
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

# 상위 폴더를 path에 추가 (프로젝트 루트)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 같은 폴더(src)에 있으므로 직접 import
from train import train_model, find_easy_examples
from data.dataloader import load_mnli_raw
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Train with easy examples excluded, then mine again")
    
    # Stage 2 specific args
    parser.add_argument("--stage1_easy_json", type=str, required=True,
                        help="Path to stage1 easy examples JSON file")
    parser.add_argument("--stage2_model_name", type=str, default=None,
                        help="Model name for stage2 training (larger model). Required if not using --skip_training")
    parser.add_argument("--stage2_output_dir", type=str, default=None,
                        help="Output directory for stage2 model")
    parser.add_argument("--confidence_threshold", type=float, default=0.8,
                        help="Confidence threshold for easy example mining (ignored if --no_confidence_check is set)")
    parser.add_argument("--no_confidence_check", action="store_true",
                        help="Only check if prediction is correct, ignore confidence threshold")
    
    # Training args (pass-through to train_model)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
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
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and load existing model from --stage2_output_dir")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: cuda if available, else cpu)")
    
    args = parser.parse_args()
    
    # Validation
    if not args.skip_training and args.stage2_model_name is None:
        parser.error("--stage2_model_name is required when not using --skip_training")
    
    print("="*70)
    print("Stage 2: Conflicting Proxy Training Pipeline")
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
    
    # Step b) Train model with easy examples excluded (또는 기존 모델 로드)
    temp_exclude_json = None
    if args.skip_training:
        # 기존 모델 로드
        if args.stage2_output_dir is None:
            raise ValueError("--stage2_output_dir must be provided when using --skip_training")
        
        print(f"\n[Step b] Loading existing stage2 model from: {args.stage2_output_dir}")
        print(f"  Skipping training...")
        
        model = AutoModelForCausalLM.from_pretrained(args.stage2_output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.stage2_output_dir)
        
        # pad token 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"  ✓ Model loaded successfully")
        
        # train_dataset_raw는 필요 없지만 변수는 유지 (호환성)
        train_dataset_raw = None
    else:
        # 모델 훈련
        # Save exclude_ids to a temporary JSON file for train_model
        import tempfile
        temp_exclude_json = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
        json.dump([{"example_id": id} for id in exclude_ids_set], temp_exclude_json, indent=2)
        temp_exclude_json.close()
        
        print(f"\n[Step b] Training stage2 model: {args.stage2_model_name}")
        print(f"  Excluding {len(exclude_ids_set)} easy examples from training")
        
        # Generate output_dir if not provided
        if args.stage2_output_dir is None:
            model_short = args.stage2_model_name.split("/")[-1] if "/" in args.stage2_model_name else args.stage2_model_name
            args.stage2_output_dir = f"./checkpoints/stage2/{model_short}-lr{args.learning_rate}-ep{args.num_epochs}"
        
        print(f"  Output directory: {args.stage2_output_dir}")
        
        # Sanity check: verify filtering will work
        print(f"\n  [Sanity check] Loading raw training data...")
        train_raw_sample = load_mnli_raw(split="train", limit=args.train_limit if args.train_limit else 10000)
        original_size = len(train_raw_sample)
        print(f"    Original training data size: {original_size}")
        print(f"    Exclude IDs range: {min(exclude_ids_set) if exclude_ids_set else 'N/A'} to {max(exclude_ids_set) if exclude_ids_set else 'N/A'}")
        
        # Spot check: verify some exclude_ids are in range
        if exclude_ids_set:
            sample_ids = list(exclude_ids_set)[:5]
            print(f"    Sample exclude IDs: {sample_ids}")
            if max(exclude_ids_set) >= original_size:
                print(f"    WARNING: Some exclude IDs ({max(exclude_ids_set)}) >= dataset size ({original_size})")
                print(f"    This may cause issues. Proceeding anyway...")
        
        # Train with exclusions
        try:
            model, tokenizer, trainer, train_dataset_raw = train_model(
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
            )
            
            print(f"\n  ✓ Stage2 training completed")
            print(f"    Remaining training examples after exclusion: {len(train_dataset_raw)}")
            
        finally:
            # Clean up temp file
            if temp_exclude_json:
                os.unlink(temp_exclude_json.name)
    
    # Step c) Load Stage1 easy examples only (not full dataset)
    print(f"\n[Step c] Loading Stage1 easy examples for re-evaluation...")
    # 전체 train 데이터를 로드하고, Stage1 easy_examples의 example_id로 필터링
    full_train_raw = load_mnli_raw(split="train", limit=None)  # 전체 데이터 로드
    
    # Stage1 easy_examples의 example_id로 필터링
    print(f"  Filtering Stage1 easy examples (IDs: {len(exclude_ids_set)} examples)...")
    stage1_easy_dataset = full_train_raw.filter(
        lambda x, idx: idx in exclude_ids_set,
        with_indices=True
    )
    print(f"  Loaded {len(stage1_easy_dataset)} Stage1 easy examples for re-evaluation")
    
    # Step d) Find real easy examples with stage2 model (among Stage1 easy examples)
    print(f"\n[Step d] Re-evaluating Stage1 easy examples with stage2 model...")
    print(f"  Stage2 model will re-evaluate {len(stage1_easy_dataset)} examples that 70m model found easy")
    
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
        dataset=stage1_easy_dataset,  # Stage1 easy examples만 평가
        confidence_threshold=args.confidence_threshold,
        device=device,
        max_examples=None,  # 전체 Stage1 easy examples 평가
        batch_size=32,
        use_confidence=not args.no_confidence_check,
    )
    
    print(f"\n  ✓ Found {len(easy_examples_stage2)} real easy examples (stage2 model correctly predicted)")
    print(f"     Out of {len(stage1_easy_dataset)} Stage1 easy examples")
    if len(stage1_easy_dataset) > 0:
        print(f"     Success rate: {len(easy_examples_stage2)/len(stage1_easy_dataset)*100:.2f}%")
    
    # Step e) Save results
    print(f"\n[Step e] Saving real easy examples...")
    
    # 모델명 추출: stage2_model_name이 있으면 사용, 없으면 stage2_output_dir에서 추출
    if args.stage2_model_name:
        model_short = args.stage2_model_name.split("/")[-1] if "/" in args.stage2_model_name else args.stage2_model_name
    elif args.stage2_output_dir:
        # stage2_output_dir에서 모델명 추출 (예: checkpoints/stage2/pythia-160m-lr5e-05-ep1 -> pythia-160m)
        dir_name = os.path.basename(args.stage2_output_dir.rstrip("/"))
        # 하이퍼파라미터 부분 제거 (-lr부터 뒤 부분)
        if "-lr" in dir_name:
            model_short = dir_name.split("-lr")[0]
        else:
            model_short = dir_name
    else:
        model_short = "unknown_model"
    
    threshold_str = f"conf{args.confidence_threshold}" if not args.no_confidence_check else "correct_only"
    output_json = f"real_easy_examples_{threshold_str}_stage2_{model_short}.json"
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(easy_examples_stage2, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved to: {output_json}")
    
    # Summary
    print("\n" + "="*70)
    print("Stage 2 Pipeline Summary")
    print("="*70)
    print(f"Stage1 easy examples (70m model): {len(exclude_ids_set)}")
    if train_dataset_raw is not None:
        print(f"Training data (hard examples for 160m): {len(train_dataset_raw)}")
    else:
        print(f"Training data: (skipped - using existing model)")
    print(f"Stage1 easy examples re-evaluated: {len(stage1_easy_dataset)}")
    print(f"Real easy examples (160m model also correct): {len(easy_examples_stage2)}")
    if len(stage1_easy_dataset) > 0:
        print(f"Success rate: {len(easy_examples_stage2)/len(stage1_easy_dataset)*100:.2f}%")
    print(f"Output JSON: {output_json}")
    print("="*70)


if __name__ == "__main__":
    main()

