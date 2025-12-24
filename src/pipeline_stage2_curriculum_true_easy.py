"""
Stage 2: Curriculum Learning + True-Easy Mining Pipeline

이 스크립트는 curriculum learning 방식으로 학습한 후,
hard 학습 전후 easy set의 변화를 추적하여 "true-easy" 예제를 선별합니다.

Phase 1: Stage1 easy set만으로 학습
Phase 2: Stage1 easy를 제외한 hard set으로 이어서 학습
True-Easy Mining: Phase 2 학습 후에도 안정적으로 쉬운 예제를 선별

How to run:
    python src/pipeline_stage2_curriculum_true_easy.py \
        --stage1_easy_json ./easy_examples_confidence_0.8_1_5e-05.json \
        --model_name EleutherAI/pythia-160m \
        --output_dir ./checkpoints/stage2_curriculum \
        --num_epochs_easy 1 \
        --num_epochs_hard 1 \
        --lr_easy 5e-5 \
        --lr_hard 5e-5 \
        --true_easy_min_conf_after 0.7 \
        --true_easy_max_conf_drop 0.1 \
        --true_easy_max_loss_increase 0.5
"""

import os
import sys
import json
import argparse
import torch
import numpy as np

# 상위 폴더를 path에 추가 (프로젝트 루트)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 같은 폴더(src)에 있으므로 직접 import
from train import train_model, evaluate_examples
from data.dataloader import load_mnli_raw
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Curriculum Learning + True-Easy Mining"
    )
    
    # Stage 2 specific args
    parser.add_argument("--stage1_easy_json", type=str, required=True,
                        help="Path to stage1 easy examples JSON file")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Base model name for stage2 training")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for stage2 models (will create phase1_easy/ and phase2_hard/ subdirs)")
    
    # Training args - Phase 1 (Easy)
    parser.add_argument("--num_epochs_easy", type=int, default=1,
                        help="Number of epochs for Phase 1 (easy set)")
    parser.add_argument("--lr_easy", type=float, default=5e-5,
                        help="Learning rate for Phase 1")
    
    # Training args - Phase 2 (Hard)
    parser.add_argument("--num_epochs_hard", type=int, default=1,
                        help="Number of epochs for Phase 2 (hard set)")
    parser.add_argument("--lr_hard", type=float, default=5e-5,
                        help="Learning rate for Phase 2")
    
    # Common training args
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--tokenize_batch_size", type=int, default=1000)
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["steps", "epoch"])
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=None)
    
    # Evaluation/debug args
    parser.add_argument("--eval_limit_easyset", type=int, default=None,
                        help="Limit number of easy set examples for evaluation (for debugging)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: cuda if available, else cpu)")
    
    # True-easy thresholds
    parser.add_argument("--true_easy_min_conf_after", type=float, default=0.7,
                        help="Minimum confidence (true_prob) after Phase 2")
    parser.add_argument("--true_easy_max_conf_drop", type=float, default=0.1,
                        help="Maximum confidence drop (before - after)")
    parser.add_argument("--true_easy_max_loss_increase", type=float, default=0.5,
                        help="Maximum loss increase (nll_after - nll_before)")
    parser.add_argument("--require_correct_before_after", action="store_true",
                        help="Require correct prediction both before and after Phase 2")
    
    # Other args
    parser.add_argument("--exclude_id_field", type=str, default="example_id",
                        help="Field name in JSON for example IDs")
    
    args = parser.parse_args()
    
    # Device 설정
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*70)
    print("Stage 2: Curriculum Learning + True-Easy Mining Pipeline")
    print("="*70)
    
    # Step A: Stage1 easy JSON 로드 및 example_id 수집
    print(f"\n[Step A] Loading stage1 easy examples from: {args.stage1_easy_json}")
    with open(args.stage1_easy_json, "r", encoding="utf-8") as f:
        stage1_easy_data = json.load(f)
    
    if not isinstance(stage1_easy_data, list):
        raise ValueError(f"Expected list in JSON, got {type(stage1_easy_data)}")
    
    if len(stage1_easy_data) == 0:
        raise ValueError("Stage1 easy JSON is empty")
    
    # example_id 수집 및 proxy_confidence 매핑 저장
    easy_ids_set = set()
    proxy_confidence_map = {}  # example_id -> proxy_confidence (stage1 모델의 confidence)
    
    if isinstance(stage1_easy_data[0], dict):
        for item in stage1_easy_data:
            example_id = item.get(args.exclude_id_field)
            if example_id is not None:
                easy_ids_set.add(example_id)
                # proxy_confidence 저장 (있으면)
                if "confidence" in item:
                    proxy_confidence_map[example_id] = item["confidence"]
    else:
        # 단순 ID 리스트인 경우
        easy_ids_set = set(stage1_easy_data)
    
    print(f"  Found {len(easy_ids_set)} unique example IDs")
    print(f"  Found {len(proxy_confidence_map)} examples with proxy_confidence")
    
    # Step B: MNLI train 전체 로드 후 easy/hard set 구성
    print(f"\n[Step B] Loading full MNLI training data and splitting into easy/hard sets...")
    full_train_raw = load_mnli_raw(split="train", limit=None)
    
    # 범위 체크
    max_id = max(easy_ids_set) if easy_ids_set else -1
    if max_id >= len(full_train_raw):
        print(f"  WARNING: max(example_id)={max_id} >= dataset_size={len(full_train_raw)}")
        print(f"  This may cause issues. Proceeding anyway...")
    
    print(f"  Full training data size: {len(full_train_raw)}")
    
    # easy_set: select() + map()으로 example_id 컬럼을 명시적으로 추가
    sorted_easy_ids = sorted(list(easy_ids_set))
    easy_set = full_train_raw.select(sorted_easy_ids)
    easy_set = easy_set.map(
        lambda x, idx: {"example_id": sorted_easy_ids[idx]},
        with_indices=True
    )
    
    # hard_set: 나머지 예제들
    hard_ids = [i for i in range(len(full_train_raw)) if i not in easy_ids_set]
    hard_set = full_train_raw.select(hard_ids)
    hard_set = hard_set.map(
        lambda x, idx: {"example_id": hard_ids[idx]},
        with_indices=True
    )
    
    print(f"  Easy set size: {len(easy_set)}")
    print(f"  Hard set size: {len(hard_set)}")
    
    # Step C: Phase 1 학습 (easy_set만)
    print(f"\n[Step C] Phase 1: Training on easy set only...")
    phase1_output_dir = os.path.join(args.output_dir, "phase1_easy")
    os.makedirs(phase1_output_dir, exist_ok=True)
    
    print(f"  Output directory: {phase1_output_dir}")
    print(f"  Training on {len(easy_set)} easy examples")
    print(f"  Epochs: {args.num_epochs_easy}, LR: {args.lr_easy}")
    
    model_phase1, tokenizer_phase1, trainer_phase1, _ = train_model(
        model_name=args.model_name,
        output_dir=phase1_output_dir,
        num_train_epochs=args.num_epochs_easy,
        batch_size=args.batch_size,
        learning_rate=args.lr_easy,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        num_proc=args.num_proc,
        tokenize_batch_size=args.tokenize_batch_size,
        save_total_limit=args.save_total_limit,
        train_raw_override=easy_set,  # easy_set만 사용
        eval_raw_override=easy_set.select(range(min(1000, len(easy_set)))) if len(easy_set) > 0 else easy_set,  # validation도 easy_set 일부 사용
    )
    
    print(f"  ✓ Phase 1 training completed")
    
    # Step D: Phase 1 이후, easy_set 평가값(before) 저장
    print(f"\n[Step D] Evaluating easy set after Phase 1 (before Phase 2)...")
    
    # eval_easy_set: easy_set에서 평가할 부분만 선택
    # select()는 컬럼을 유지하므로 example_id도 그대로 유지됨
    eval_easy_set = easy_set
    if args.eval_limit_easyset:
        eval_easy_set = easy_set.select(range(min(args.eval_limit_easyset, len(easy_set))))
    
    print(f"  Evaluating {len(eval_easy_set)} examples from easy set...")
    
    model_phase1.to(device)
    before_results = evaluate_examples(
        model=model_phase1,
        tokenizer=tokenizer_phase1,
        dataset=eval_easy_set,
        device=device,
        batch_size=32,
        max_examples=None,
    )
    
    # before_map 구성: example_id -> stats
    # evaluate_examples()가 dataset의 example_id 컬럼을 읽어서 반환하므로 변환 불필요
    before_map = {result["example_id"]: result for result in before_results}
    
    print(f"  ✓ Evaluated {len(before_map)} examples")
    
    # Step E: Phase 2 학습 (hard_set로 이어서 학습)
    print(f"\n[Step E] Phase 2: Continuing training on hard set...")
    phase2_output_dir = os.path.join(args.output_dir, "phase2_hard")
    os.makedirs(phase2_output_dir, exist_ok=True)
    
    print(f"  Output directory: {phase2_output_dir}")
    print(f"  Continuing from Phase 1 checkpoint: {phase1_output_dir}")
    print(f"  Training on {len(hard_set)} hard examples")
    print(f"  Epochs: {args.num_epochs_hard}, LR: {args.lr_hard}")
    
    # Phase1 체크포인트에서 이어서 학습
    model_phase2, tokenizer_phase2, trainer_phase2, _ = train_model(
        model_name=phase1_output_dir,  # Phase1 체크포인트에서 시작
        output_dir=phase2_output_dir,
        num_train_epochs=args.num_epochs_hard,
        batch_size=args.batch_size,
        learning_rate=args.lr_hard,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        num_proc=args.num_proc,
        tokenize_batch_size=args.tokenize_batch_size,
        save_total_limit=args.save_total_limit,
        train_raw_override=hard_set,  # hard_set만 사용
        eval_raw_override=hard_set.select(range(min(1000, len(hard_set)))) if len(hard_set) > 0 else hard_set,  # validation도 hard_set 일부 사용
    )
    
    print(f"  ✓ Phase 2 training completed")
    
    # Step F: Phase 2 이후, easy_set 평가값(after) 저장
    print(f"\n[Step F] Evaluating easy set after Phase 2...")
    
    print(f"  Evaluating {len(eval_easy_set)} examples from easy set...")
    
    model_phase2.to(device)
    after_results = evaluate_examples(
        model=model_phase2,
        tokenizer=tokenizer_phase2,
        dataset=eval_easy_set,
        device=device,
        batch_size=32,
        max_examples=None,
    )
    
    # after_map 구성: example_id -> stats
    # evaluate_examples()가 dataset의 example_id 컬럼을 읽어서 반환하므로 변환 불필요
    after_map = {result["example_id"]: result for result in after_results}
    
    print(f"  ✓ Evaluated {len(after_map)} examples")
    
    # Step G: True-Easy 선별 로직
    print(f"\n[Step G] Selecting true-easy examples based on stability...")
    print(f"  Thresholds:")
    print(f"    min_conf_after: {args.true_easy_min_conf_after}")
    print(f"    max_conf_drop: {args.true_easy_max_conf_drop}")
    print(f"    max_loss_increase: {args.true_easy_max_loss_increase}")
    print(f"    require_correct_before_after: {args.require_correct_before_after}")
    
    true_easy = []
    
    # 모든 example_id에 대해 비교
    all_ids = set(before_map.keys()) & set(after_map.keys())
    print(f"  Comparing {len(all_ids)} examples that were evaluated in both phases...")
    
    for example_id in all_ids:
        before_stats = before_map[example_id]
        after_stats = after_map[example_id]
        
        # 변화량 계산
        conf_drop = before_stats["true_prob"] - after_stats["true_prob"]
        loss_increase = after_stats["nll"] - before_stats["nll"]
        
        # True-Easy 조건 체크
        is_true_easy = True
        
        # 1. after confidence가 최소값 이상인지
        if after_stats["true_prob"] < args.true_easy_min_conf_after:
            is_true_easy = False
        
        # 2. confidence drop이 최대값 이하인지
        if conf_drop > args.true_easy_max_conf_drop:
            is_true_easy = False
        
        # 3. loss increase가 최대값 이하인지
        if loss_increase > args.true_easy_max_loss_increase:
            is_true_easy = False
        
        # 4. before/after 모두 correct인지 (옵션)
        if args.require_correct_before_after:
            if not (before_stats["correct"] and after_stats["correct"]):
                is_true_easy = False
        
        if is_true_easy:
            # 원본 예제 데이터 가져오기
            # evaluate_examples에서 반환된 example_id는 원본 full_train_raw의 인덱스
            example_data = None
            if example_id < len(full_train_raw):
                example_data = full_train_raw[example_id]
            
            true_easy_item = {
                "example_id": example_id,
                "proxy_confidence": proxy_confidence_map.get(example_id, None),
                "before": {
                    "true_prob": before_stats["true_prob"],
                    "nll": before_stats["nll"],
                    "correct": before_stats["correct"],
                    "predicted_label": before_stats["predicted_label"],
                },
                "after": {
                    "true_prob": after_stats["true_prob"],
                    "nll": after_stats["nll"],
                    "correct": after_stats["correct"],
                    "predicted_label": after_stats["predicted_label"],
                },
                "conf_drop": conf_drop,
                "loss_increase": loss_increase,
                "true_label": before_stats["true_label"],
            }
            
            # 원본 데이터가 있으면 추가
            if example_data:
                true_easy_item["premise"] = example_data.get("premise")
                true_easy_item["hypothesis"] = example_data.get("hypothesis")
            
            true_easy.append(true_easy_item)
    
    print(f"  ✓ Found {len(true_easy)} true-easy examples")
    if len(all_ids) > 0:
        print(f"    Success rate: {len(true_easy)/len(all_ids)*100:.2f}%")
    
    # Step H: 결과 저장
    print(f"\n[Step H] Saving true-easy examples...")
    
    # 파일명 생성 (threshold 정보 포함)
    model_short = args.model_name.split("/")[-1] if "/" in args.model_name else args.model_name
    threshold_str = (
        f"conf{args.true_easy_min_conf_after}_"
        f"drop{args.true_easy_max_conf_drop}_"
        f"loss{args.true_easy_max_loss_increase}"
    )
    if args.require_correct_before_after:
        threshold_str += "_correct_req"
    
    output_json = f"true_easy_examples_{threshold_str}_stage2_curriculum_{model_short}.json"
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(true_easy, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved to: {output_json}")
    
    # Summary
    print("\n" + "="*70)
    print("Stage 2 Curriculum Pipeline Summary")
    print("="*70)
    print(f"Stage1 easy examples: {len(easy_ids_set)}")
    print(f"Easy set size: {len(easy_set)}")
    print(f"Hard set size: {len(hard_set)}")
    print(f"Phase 1 (easy) training: {args.num_epochs_easy} epochs, LR={args.lr_easy}")
    print(f"Phase 2 (hard) training: {args.num_epochs_hard} epochs, LR={args.lr_hard}")
    print(f"Examples evaluated: {len(all_ids)}")
    print(f"True-easy examples found: {len(true_easy)}")
    if len(all_ids) > 0:
        print(f"Success rate: {len(true_easy)/len(all_ids)*100:.2f}%")
    print(f"Output JSON: {output_json}")
    print(f"Phase 1 checkpoint: {phase1_output_dir}")
    print(f"Phase 2 checkpoint: {phase2_output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

