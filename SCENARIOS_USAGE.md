# 3가지 시나리오 실행 가이드

## 개요

이 문서는 `train_and_evaluate.py`의 3가지 시나리오를 실행하는 방법을 설명합니다.

각 시나리오는 Qwen/Qwen2.5-0.5B 모델을 사용하며, Qwen2.5 표준 하이퍼파라미터로 훈련됩니다:
- **Batch size per device**: 8
- **Gradient accumulation steps**: 2 (Effective batch size = 16)
- **Learning rate**: 2e-5
- **Optimizer**: AdamW (beta1=0.9, beta2=0.999, epsilon=1e-6)
- **Weight decay**: 0.01
- **Scheduler**: Cosine
- **Gradient clipping**: 1.0

---

## Scenario 1: Full MNLI Training

전체 MNLI training 데이터로 훈련하는 기본 시나리오입니다.

### 실행 명령어

```bash
python src/train_and_evaluate.py \
    --scenario 1 \
    --output_dir ./checkpoints/scenario1_full \
    --num_epochs 3 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --num_proc 16 \
    --tokenize_batch_size 1000 \
    --eval_strategy epoch \
    --save_total_limit 2 \
    --eval_mnli_dev \
    --eval_hans
```

### 최소 명령어 (기본값 사용)

```bash
python src/train_and_evaluate.py \
    --scenario 1 \
    --output_dir ./checkpoints/scenario1_full
```

### 출력
- 모델 체크포인트: `./checkpoints/scenario1_full/full_mnli/`
- 평가 결과: `./checkpoints/scenario1_full/full_mnli/evaluation_results.json`

---

## Scenario 2: True-Easy (1 epoch) → Full MNLI

True-easy 예제로 1 epoch 훈련한 후, 전체 MNLI 데이터로 이어서 훈련합니다.

### 실행 명령어

```bash
python src/train_and_evaluate.py \
    --scenario 2 \
    --output_dir ./checkpoints/scenario2_true_easy_full \
    --true_easy_json /home/user3/data-analysis/true_easy_examples_conf0.4_drop0.1_loss0.5_stage2_curriculum_pythia-410m.json \
    --num_epochs 3 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --num_proc 16 \
    --tokenize_batch_size 1000 \
    --eval_strategy epoch \
    --save_total_limit 2 \
    --eval_mnli_dev \
    --eval_hans
```

### 최소 명령어 (기본값 사용)

```bash
python src/train_and_evaluate.py \
    --scenario 2 \
    --output_dir ./checkpoints/scenario2_true_easy_full \
    --true_easy_json /home/user3/data-analysis/true_easy_examples_conf0.4_drop0.1_loss0.5_stage2_curriculum_pythia-410m.json
```

### 출력
- Phase 1 모델: `./checkpoints/scenario2_true_easy_full/scenario2_phase1_true_easy/`
- Phase 2 모델 (최종): `./checkpoints/scenario2_true_easy_full/scenario2_phase2_full/`
- 평가 결과: `./checkpoints/scenario2_true_easy_full/scenario2_phase2_full/evaluation_results.json`

### 훈련 단계
1. **Phase 1**: True-easy set으로 1 epoch 훈련
2. **Phase 2**: Phase 1 체크포인트에서 이어서 전체 MNLI 데이터로 훈련 (3 epochs)
3. **평가**: Phase 2 모델로 MNLI dev와 HANS 평가

---

## Scenario 3: True-Easy → Easy → Hard (Curriculum, No Overlap)

데이터가 겹치지 않도록 True-easy → Easy → Hard 순서로 curriculum learning을 수행합니다.

### 데이터 분할
- **True-easy set**: True-easy JSON의 예제들
- **Easy set**: Stage1 easy JSON의 예제들 중 True-easy에 포함되지 않은 것들
- **Hard set**: 나머지 모든 예제들

### 실행 명령어

```bash
python src/train_and_evaluate.py \
    --scenario 3 \
    --output_dir ./checkpoints/scenario3_curriculum_3stage \
    --true_easy_json /home/user3/data-analysis/true_easy_examples_conf0.4_drop0.1_loss0.5_stage2_curriculum_pythia-410m.json \
    --stage1_easy_json /home/user3/data-analysis/easy_examples_confidence_pythia-160m_0.8_3_5e-05.json \
    --num_epochs_easy 1 \
    --num_epochs_hard 2 \
    --lr_easy 2e-5 \
    --lr_hard 2e-5 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_proc 16 \
    --tokenize_batch_size 1000 \
    --eval_strategy epoch \
    --save_total_limit 2 \
    --eval_mnli_dev \
    --eval_hans
```

### 최소 명령어 (기본값 사용)

```bash
python src/train_and_evaluate.py \
    --scenario 3 \
    --output_dir ./checkpoints/scenario3_curriculum_3stage \
    --true_easy_json /home/user3/data-analysis/true_easy_examples_conf0.4_drop0.1_loss0.5_stage2_curriculum_pythia-410m.json \
    --stage1_easy_json /home/user3/data-analysis/easy_examples_confidence_pythia-160m_0.8_3_5e-05.json
```

### 출력
- Phase 1 모델: `./checkpoints/scenario3_curriculum_3stage/scenario3_phase1_true_easy/`
- Phase 2 모델: `./checkpoints/scenario3_curriculum_3stage/scenario3_phase2_easy/`
- Phase 3 모델 (최종): `./checkpoints/scenario3_curriculum_3stage/scenario3_phase3_hard/`
- 평가 결과: `./checkpoints/scenario3_curriculum_3stage/scenario3_phase3_hard/evaluation_results.json`

### 훈련 단계
1. **Phase 1**: True-easy set으로 훈련 (`--num_epochs_easy` epochs)
2. **Phase 2**: Phase 1 체크포인트에서 Easy set으로 이어서 훈련 (`--num_epochs_easy` epochs)
3. **Phase 3**: Phase 2 체크포인트에서 Hard set으로 이어서 훈련 (`--num_epochs_hard` epochs)
4. **평가**: Phase 3 모델로 MNLI dev와 HANS 평가

---

## 주요 옵션 설명

### 필수 옵션
- `--scenario`: 시나리오 번호 (1, 2, 또는 3)
- `--output_dir`: 결과를 저장할 디렉토리

### 시나리오별 필수 옵션
- **Scenario 2, 3**: `--true_easy_json` (True-easy 예제 JSON 경로)
- **Scenario 3**: `--stage1_easy_json` (Stage1 easy 예제 JSON 경로)

### 훈련 옵션
- `--num_epochs`: 전체 훈련 epoch 수 (기본값: 3)
- `--batch_size`: Device당 배치 크기 (기본값: 8)
- `--gradient_accumulation_steps`: Gradient accumulation steps (기본값: 2)
  - Effective batch size = batch_size × gradient_accumulation_steps = 16
- `--learning_rate`: 학습률 (기본값: 2e-5)
- `--num_proc`: 토크나이징 프로세스 수 (기본값: 16)
- `--eval_strategy`: 평가 전략 (`epoch` 또는 `steps`, 기본값: `epoch`)
- `--save_total_limit`: 저장할 최대 체크포인트 수 (기본값: None, 모두 저장)

### 시나리오 3 전용 옵션
- `--num_epochs_easy`: Easy set 훈련 epoch 수 (기본값: 1)
- `--num_epochs_hard`: Hard set 훈련 epoch 수 (기본값: 1)
- `--lr_easy`: Easy set 학습률 (기본값: 2e-5)
- `--lr_hard`: Hard set 학습률 (기본값: 2e-5)

### 평가 옵션
- `--eval_mnli_dev`: MNLI dev 평가 수행 (기본값: True)
- `--eval_hans`: HANS 평가 수행 (기본값: True)
- `--eval_batch_size`: 평가 배치 크기 (기본값: 32)
- `--device`: 사용할 디바이스 (기본값: 자동 감지)

---

## 빠른 시작 (기본값 사용)

```bash
# Scenario 1
python src/train_and_evaluate.py --scenario 1 --output_dir ./checkpoints/scenario1

# Scenario 2
python src/train_and_evaluate.py \
    --scenario 2 \
    --output_dir ./checkpoints/scenario2 \
    --true_easy_json /path/to/true_easy_examples.json

# Scenario 3
python src/train_and_evaluate.py \
    --scenario 3 \
    --output_dir ./checkpoints/scenario3 \
    --true_easy_json /path/to/true_easy_examples.json \
    --stage1_easy_json /path/to/stage1_easy_examples.json
```

---

## 출력 파일 구조

각 시나리오 실행 후 다음과 같은 파일들이 생성됩니다:

```
checkpoints/
├── scenario1_full/
│   └── full_mnli/
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer_config.json
│       └── evaluation_results.json
│
├── scenario2_true_easy_full/
│   ├── scenario2_phase1_true_easy/
│   │   └── ...
│   └── scenario2_phase2_full/
│       ├── ...
│       └── evaluation_results.json
│
└── scenario3_curriculum_3stage/
    ├── scenario3_phase1_true_easy/
    │   └── ...
    ├── scenario3_phase2_easy/
    │   └── ...
    └── scenario3_phase3_hard/
        ├── ...
        └── evaluation_results.json
```

---

## 참고사항

1. **GPU 메모리**: Qwen2.5-0.5B 모델은 배치 사이즈 8과 gradient accumulation 2로 메모리를 효율적으로 사용합니다.

2. **모델 경로**: 기본 모델은 `Qwen/Qwen2.5-0.5B`입니다. 다른 모델을 사용하려면 `--model_name` 옵션을 지정하세요.

3. **평가 스킵**: 평가를 스킵하고 훈련만 수행하려면 `--eval_mnli_dev`와 `--eval_hans`를 제거하세요.

4. **디버깅**: 작은 데이터셋으로 테스트하려면 훈련 옵션은 그대로 두고, JSON 파일을 작은 샘플로 만들어 테스트할 수 있습니다.

