# Easy Example Analysis Pipeline

쉬운 예제 판별 기준 2가지를 구현하고, 각 기준으로 선별된 easy set 안에 hypothesis-only 편향 예제가 얼마나 포함되는지 정량화하여 시각화하는 파이프라인입니다.

## 목표

- **정의 A (단일 모델 스냅샷 기반)**: 고정 모델에서 loss가 낮은 샘플을 easy로 정의
- **정의 B (학습 진행 기반)**: 학습 전후 loss 감소량이 큰 샘플을 easy로 정의
- **편향 분석**: 각 easy set에서 hypothesis-only 편향 예제의 비율 측정
- **시각화**: Figure 1 생성 (2패널 비교)

## 요구사항

```bash
pip install torch transformers datasets tqdm numpy pandas matplotlib pyarrow
```

## 파이프라인 구조

```
preliminary/script1_definition_a.py  → Definition A 점수 계산 및 easy set 생성
preliminary/script2_definition_b.py  → Definition B 점수 계산 (학습 포함) 및 easy set 생성
preliminary/script3_hypothesis_bias.py → Hypothesis-only 편향 태깅
preliminary/script4_merge_and_aggregate.py → 결과 병합 및 bias rate 집계
preliminary/script5_plot_figure1.py  → Figure 1 생성
```

## 빠른 시작

### 전체 파이프라인 실행

```bash
# 실행 스크립트 수정 (체크포인트 경로 설정)
vim run_all_scripts.sh

# 전체 파이프라인 실행
./run_all_scripts.sh
```

### 단계별 실행

#### Step 1: Definition A (단일 모델 스냅샷 기반)

이미 훈련된 체크포인트(예: Pythia-160m, 1 epoch)를 사용하여 모든 샘플의 loss를 계산합니다.

```bash
python preliminary/script1_definition_a.py \
    --checkpoint_path ./checkpoints/pythia-160m-lr5e-05-ep1 \
    --output_dir ./results/definition_a \
    --top_k_percent 5 10 20 40 \
    --batch_size 32 \
    --seed 42
```

**출력**:
- `definition_a_scores.jsonl`: 각 샘플의 loss 점수
- `easy_set_top_{k}_percent.json`: 각 k%에 대한 easy set ID 리스트
- `metadata.json`: 설정 정보

#### Step 2: Definition B (학습 진행 기반)

Base 모델로 loss를 계산한 후, 짧게 학습시킨 모델로 다시 계산하여 learning speed를 측정합니다.

```bash
python preliminary/script2_definition_b.py \
    --base_model_name EleutherAI/pythia-160m \
    --output_dir ./results/definition_b \
    --training_epochs 1 \
    --training_lr 5e-5 \
    --training_batch_size 16 \
    --top_k_percent 5 10 20 40 \
    --eval_batch_size 32 \
    --seed 42
```

**옵션**: 이미 학습된 체크포인트가 있다면 `--skip_training` 사용:
```bash
python preliminary/script2_definition_b.py \
    --base_model_name EleutherAI/pythia-160m \
    --output_dir ./results/definition_b \
    --skip_training \
    --trained_checkpoint ./checkpoints/pythia-160m-lr5e-05-ep1 \
    --top_k_percent 5 10 20 40 \
    --eval_batch_size 32 \
    --seed 42
```

**출력**:
- `definition_b_scores.jsonl`: 각 샘플의 learning speed (delta_loss)
- `easy_set_top_{k}_percent.json`: 각 k%에 대한 easy set ID 리스트
- `metadata.json`: 설정 정보

#### Step 3: Hypothesis-only 편향 태깅

Premise 없이 hypothesis만 사용하여 평가하고, 정답을 맞추는 샘플에 bias 태그를 부여합니다.

```bash
python preliminary/script3_hypothesis_bias.py \
    --checkpoint_path ./checkpoints/pythia-160m-lr5e-05-ep3 \
    --output_file ./results/hypothesis_bias_tags.jsonl \
    --batch_size 32 \
    --seed 42
```

**출력**:
- `hypothesis_bias_tags.jsonl`: 각 샘플의 bias 태그 (0 또는 1)

#### Step 4: 결과 병합 및 Bias Rate 집계

정의 A, 정의 B 점수와 bias 태그를 병합하여 각 easy set의 bias rate를 계산합니다.

```bash
python preliminary/script4_merge_and_aggregate.py \
    --definition_a_scores ./results/definition_a/definition_a_scores.jsonl \
    --definition_b_scores ./results/definition_b/definition_b_scores.jsonl \
    --bias_tags ./results/hypothesis_bias_tags.jsonl \
    --definition_a_dir ./results/definition_a \
    --definition_b_dir ./results/definition_b \
    --output_dir ./results/merged \
    --top_k_percent 5 10 20 40
```

**출력**:
- `merged_scores.jsonl`: 병합된 점수 및 bias 태그
- `merged_scores.csv`, `merged_scores.parquet`: 분석용 테이블
- `bias_rates.json`: 각 easy set의 bias rate
- `bias_rates_summary.csv`: 요약 테이블 (enrichment ratio 포함)

#### Step 5: Figure 1 생성

정의 A와 정의 B의 bias rate 결과를 시각화합니다.

```bash
python preliminary/script5_plot_figure1.py \
    --bias_rates_file ./results/merged/bias_rates.json \
    --output_file ./figures/fig1.png
```

**출력**:
- `fig1.png`: Figure 1 (PNG, 300 DPI)
- `fig1.pdf`: Figure 1 (PDF)

## 출력 파일 구조

```
results/
├── definition_a/
│   ├── definition_a_scores.jsonl
│   ├── easy_set_top_5_percent.json
│   ├── easy_set_top_10_percent.json
│   ├── easy_set_top_20_percent.json
│   ├── easy_set_top_40_percent.json
│   └── metadata.json
│
├── definition_b/
│   ├── definition_b_scores.jsonl
│   ├── easy_set_top_5_percent.json
│   ├── easy_set_top_10_percent.json
│   ├── easy_set_top_20_percent.json
│   ├── easy_set_top_40_percent.json
│   ├── trained_checkpoint/  (학습된 모델)
│   └── metadata.json
│
├── hypothesis_bias_tags.jsonl
│
└── merged/
    ├── merged_scores.jsonl
    ├── merged_scores.csv
    ├── merged_scores.parquet
    ├── bias_rates.json
    └── bias_rates_summary.csv

figures/
└── fig1.png
└── fig1.pdf
```

## 데이터 형식

### 점수 파일 (JSONL)

각 줄은 하나의 샘플을 나타냅니다:

**Definition A**:
```json
{"example_id": 0, "snapshot_loss": 0.1234, "true_label": "entailment", "predicted_label": "entailment", "correct": true, "true_prob": 0.89}
```

**Definition B**:
```json
{"example_id": 0, "loss_before": 1.234, "loss_after": 0.567, "delta_loss": 0.667, "true_label": "entailment", "predicted_label": "entailment", "correct": true}
```

**Bias Tags**:
```json
{"example_id": 0, "true_label": "entailment", "predicted_label": "entailment", "correct": true, "hypothesis_only_bias": 1}
```

### Easy Set 파일 (JSON)

각 easy set은 ID 리스트입니다:
```json
[0, 5, 12, 34, ...]
```

### Bias Rates 파일 (JSON)

```json
{
  "bias_rates": {
    "definition_a": {
      "top_5": {"easy_set_size": 19635, "bias_count": 15000, "bias_rate": 0.764},
      ...
    },
    "definition_b": {
      ...
    }
  },
  "random_baseline": {
    "overall_bias_rate": 0.523,
    "total_examples": 392702,
    "total_bias_count": 205283
  },
  "top_k_percent": [5, 10, 20, 40]
}
```

## 주요 설정

### 모델 및 체크포인트

- **Definition A 체크포인트**: Pythia-160m, learning rate 5e-5, 1 epoch
- **Definition B base 모델**: EleutherAI/pythia-160m
- **Definition B 학습**: 1 epoch, learning rate 5e-5
- **Bias 태깅 체크포인트**: Pythia-160m, learning rate 5e-5, 3 epochs

### 하이퍼파라미터

- **top_k_percent**: 5, 10, 20, 40 (기본값)
- **batch_size**: 32 (평가용), 16 (학습용)
- **seed**: 42 (재현성)

## 주의사항

1. **프롬프트 일관성**: 모든 스크립트에서 동일한 프롬프트 및 토크나이징 규칙을 사용합니다.
2. **라벨 정규화**: 모든 점수 계산에서 라벨 텍스트는 공백 포함 여부까지 고정됩니다.
3. **재현성**: seed를 고정하여 결과의 재현성을 보장합니다.
4. **메모리**: 큰 모델이나 배치 사이즈를 사용할 때 GPU 메모리를 확인하세요.

## 문제 해결

### 메모리 부족
- `--batch_size`를 줄이세요 (예: 16 또는 8)
- `--training_batch_size`를 줄이세요

### 학습 시간이 너무 길어요
- `--skip_training` 옵션을 사용하여 이미 학습된 체크포인트를 재사용하세요
- `--training_epochs`를 줄이세요 (예: 1 epoch)

### 결과가 재현되지 않아요
- 모든 스크립트에서 `--seed 42`를 사용하세요
- 동일한 체크포인트를 사용하세요

## 라이선스

이 코드는 연구 목적으로 제공됩니다.

