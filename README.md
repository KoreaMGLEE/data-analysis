# MNLI 쉬운 예제 식별 연구

이 프로젝트는 MNLI 데이터셋에서 작은 모델(Pythia 30M)을 훈련하고, 높은 확신으로 정답을 맞추는 쉬운 예제를 식별합니다.

## 사용 방법

### 기본 실행 (전체 파이프라인)

```bash
python src/train.py
```

### 주요 옵션

```bash
python src/train.py \
    --model_name EleutherAI/pythia-30m \
    --output_dir ./checkpoints/pythia-30m-mnli \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --confidence_threshold 0.8 \
    --eval_limit 1000 \
    --device cuda
```

### 기존 모델로만 평가 (훈련 건너뛰기)

```bash
python src/train.py \
    --skip_training \
    --checkpoint_path ./checkpoints/pythia-30m-mnli \
    --confidence_threshold 0.8 \
    --eval_limit 1000
```

### 디버깅용 (작은 데이터셋으로 테스트)

```bash
python src/train.py \
    --train_limit 1000 \
    --eval_limit 100 \
    --num_epochs 1
```

## 주요 기능

1. **MNLI 데이터 로드**: `data/dataloader.py`를 사용하여 데이터 로드
2. **모델 훈련**: Pythia 30M 모델을 MNLI 데이터로 fine-tuning
3. **쉬운 예제 식별**: confidence 0.8 이상으로 정답을 맞추는 예제 찾기

## 출력 파일

- `./easy_examples_confidence_0.8.json`: 식별된 쉬운 예제들이 저장됩니다.
- `./checkpoints/pythia-30m-mnli/`: 훈련된 모델이 저장됩니다.

## 필요한 패키지

```bash
pip install torch transformers datasets tqdm numpy
```

