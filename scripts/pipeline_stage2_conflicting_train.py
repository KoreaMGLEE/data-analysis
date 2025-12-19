"""
Stage 2: Conflicting Proxy Training Pipeline

이 스크립트는 stage1에서 찾은 easy examples를 제외하고 더 큰 모델로 학습한 후,
학습된 모델로 다시 easy example mining을 수행합니다.

How to run:
    python src/pipeline_stage2_conflicting_train.py \
        --stage1_easy_json ./easy_examples_confidence_0.8_1_5e-05.json \
        --stage2_model_name EleutherAI/pythia-160m \
        --stage2_output_dir ./checkpoints/stage2-pythia-160m \
        --confidence_threshold 0.8 \
        --num_epochs 1 \
        --batch_size 16 \
        --learning_rate 5e-5 \
        --train_limit 2000 \
        --eval_limit 200

Smoke test:
    python src/pipeline_stage2_conflicting_train.py \
        --stage1_easy_json <path_to_easy_json> \
        --stage2_model_name EleutherAI/pythia-14m \
        --confidence_threshold 0.8 \
        --train_limit 2000 \
        --eval_limit 200 \
        --num_epochs 1
"""

import os
import sys
import json
import argparse
import torch

# 상위 폴더를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import train_model, find_easy_examples
from data.dataloader import load_mnli_raw


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Train with easy examples excluded, then mine again")
    
    # Stage 2 specific args
    parser.add_argument("--stage1_easy_json", type=str, required=True,
                        help="Path to stage1 easy examples JSON file")
    parser.add_argument("--stage2_model_name", type=str, required=True,
                        help="Model name for stage2 training (larger model)")
    parser.add_argument("--stage2_output_dir", type=str, default=None,
                        help="Output directory for stage2 model")
    parser.add_argument("--confidence_threshold", type=float, default=0.8,
                        help="Confidence threshold for easy example mining")
    
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
    
    args = parser.parse_args()
    
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
    
    # Save exclude_ids to a temporary JSON file for train_model
    import tempfile
    temp_exclude_json = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
    json.dump([{"example_id": id} for id in exclude_ids_set], temp_exclude_json, indent=2)
    temp_exclude_json.close()
    
    # Step b) Train model with easy examples excluded
    print(f"\n[Step b] Training stage2 model: {args.stage2_model_name}")
    print(f"  Excluding {len(exclude_ids_set)} easy examples from training")
    
    # Generate output_dir if not provided
    if args.stage2_output_dir is None:
        model_short = args.stage2_model_name.split("/")[-1] if "/" in args.stage2_model_name else args.stage2_model_name
        args.stage2_output_dir = f"./checkpoints/stage2-{model_short}-lr{args.learning_rate}-ep{args.num_epochs}"
    
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
        os.unlink(temp_exclude_json.name)
    
    # Step c) Load full raw training data for easy mining
    print(f"\n[Step c] Loading full raw training data for easy mining...")
    full_train_raw = load_mnli_raw(split="train", limit=args.eval_limit)
    print(f"  Loaded {len(full_train_raw)} examples for easy mining")
    
    # Step d) Find easy examples with stage2 model
    print(f"\n[Step d] Finding easy examples with stage2 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    easy_examples_stage2 = find_easy_examples(
        model=model,
        tokenizer=tokenizer,
        dataset=full_train_raw,
        confidence_threshold=args.confidence_threshold,
        device=device,
        max_examples=args.eval_limit,
        batch_size=32,
    )
    
    print(f"\n  ✓ Found {len(easy_examples_stage2)} easy examples with stage2 model")
    
    # Step e) Save results
    print(f"\n[Step e] Saving stage2 easy examples...")
    model_short = args.stage2_model_name.split("/")[-1] if "/" in args.stage2_model_name else args.stage2_model_name
    output_json = f"true_easy_examples_conf{args.confidence_threshold}_stage2_{model_short}.json"
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(easy_examples_stage2, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved to: {output_json}")
    
    # Summary
    print("\n" + "="*70)
    print("Stage 2 Pipeline Summary")
    print("="*70)
    print(f"Stage1 easy examples excluded: {len(exclude_ids_set)}")
    print(f"Remaining training data: {len(train_dataset_raw)}")
    print(f"Stage2 easy examples found: {len(easy_examples_stage2)}")
    print(f"Output JSON: {output_json}")
    print("="*70)


if __name__ == "__main__":
    main()

