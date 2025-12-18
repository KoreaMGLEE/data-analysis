import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# MNLI label mapping (GLUE MNLI)
LABEL2TEXT = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}

TEXT2LABEL = {v: k for k, v in LABEL2TEXT.items()}

def make_prompt(premise: str, hypothesis: str) -> str:
    # full setting: use both premise and hypothesis
    return (
        "Decide the relationship between the Premise and the Hypothesis.\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "Answer:"
    )

def tokenize_fn(example, tokenizer, max_length: int = 256):
    """
    Build (prompt + target_text) and create:
      - input_ids / attention_mask for full_text
      - labels = input_ids but prompt part masked as -100 (loss only on target)
    """
    label = example.get("label", None)
    # Some splits (especially test) can have label == -1
    if label is None or label == -1:
        return {"input_ids": [], "attention_mask": [], "labels": [], "input_text": "", "target_text": ""}

    premise = example["premise"]
    hypothesis = example["hypothesis"]

    prompt = make_prompt(premise, hypothesis)
    target_text = LABEL2TEXT[int(label)]

    # The model sees prompt and is trained to generate target_text
    full_text = prompt + " " + target_text

    # Tokenize prompt separately to know where to mask
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    # Tokenize full text (prompt + answer)
    full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    # Build labels: same length as input_ids
    labels = input_ids.copy()

    # Mask prompt tokens so loss is computed only on the answer tokens
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "input_text": prompt,        # debug (string)
        "target_text": target_text,  # debug (string)
    }


def tokenize_fn_batch(examples, tokenizer, max_length: int = 256):
    """
    배치 단위로 토크나이징 (더 빠름)
    
    Args:
        examples: 딕셔너리 형태의 배치 (각 값이 리스트)
        tokenizer: 토크나이저
        max_length: 최대 시퀀스 길이
    
    Returns:
        딕셔너리 형태의 배치 결과 (각 값이 리스트)
    """
    premises = examples["premise"]
    hypotheses = examples["hypothesis"]
    labels = examples["label"]
    
    batch_size = len(premises)
    
    # 프롬프트와 전체 텍스트 생성
    prompts = []
    target_texts = []
    full_texts = []
    valid_indices = []
    
    for i, (premise, hypothesis, label) in enumerate(zip(premises, hypotheses, labels)):
        if label is None or label == -1:
            continue
        valid_indices.append(i)
        prompt = make_prompt(premise, hypothesis)
        target_text = LABEL2TEXT[int(label)]
        prompts.append(prompt)
        target_texts.append(target_text)
        full_texts.append(prompt + " " + target_text)
    
    if len(full_texts) == 0:
        # 유효한 예제가 없는 경우
        return {
            "input_ids": [[]] * batch_size,
            "attention_mask": [[]] * batch_size,
            "labels": [[]] * batch_size,
            "input_text": [""] * batch_size,
            "target_text": [""] * batch_size,
        }
    
    # 배치 토크나이징 (유효한 프롬프트와 전체 텍스트만)
    prompt_tokenized = tokenizer(
        prompts,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )
    
    full_tokenized = tokenizer(
        full_texts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    
    # 결과 초기화
    input_ids_list = [[] for _ in range(batch_size)]
    attention_mask_list = [[] for _ in range(batch_size)]
    labels_list = [[] for _ in range(batch_size)]
    input_text_list = [""] * batch_size
    target_text_list = [""] * batch_size
    
    # 각 유효한 예제 처리
    for idx, valid_idx in enumerate(valid_indices):
        prompt_ids = prompt_tokenized["input_ids"][idx]
        full_ids = full_tokenized["input_ids"][idx]
        full_mask = full_tokenized["attention_mask"][idx]
        
        # Labels 생성 및 프롬프트 부분 마스킹
        labels = full_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        
        input_ids_list[valid_idx] = full_ids
        attention_mask_list[valid_idx] = full_mask
        labels_list[valid_idx] = labels
        input_text_list[valid_idx] = prompts[idx]
        target_text_list[valid_idx] = target_texts[idx]
    
    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
        "input_text": input_text_list,
        "target_text": target_text_list,
    }

import torch

def collate_fn(batch, tokenizer):
    batch = [x for x in batch if len(x["input_ids"]) > 0]

    # 1) input_ids / attention_mask만 tokenizer.pad로 패딩
    pad_inputs = [
        {"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]}
        for x in batch
    ]
    padded = tokenizer.pad(pad_inputs, return_tensors="pt")

    # 2) labels는 우리가 직접 -100으로 패딩
    max_len = padded["input_ids"].shape[1]
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, x in enumerate(batch):
        seq = x["labels"]
        # input_ids와 동일한 길이로 만든다는 가정(우리가 tokenize_fn에서 그렇게 만들었음)
        seq_len = min(len(seq), max_len)
        labels[i, :seq_len] = torch.tensor(seq[:seq_len], dtype=torch.long)

    padded["labels"] = labels

    # 3) 디버그 문자열은 리스트로
    padded["input_text"] = [x.get("input_text", "") for x in batch]
    padded["target_text"] = [x.get("target_text", "") for x in batch]
    return padded


def load_mnli_raw(split="train", limit=None):
    """
    MNLI 데이터셋의 원본 데이터를 로드합니다.
    
    Args:
        split: 데이터셋 스플릿 ("train", "validation_matched", "validation_mismatched")
        limit: 데이터 제한 (디버깅용, None이면 전체)
    
    Returns:
        data: 원본 데이터셋 (premise, hypothesis, label 포함)
    """
    ds = load_dataset("glue", "mnli")
    data = ds[split]
    
    if limit:
        data = data.select(range(min(limit, len(data))))
    
    return data


def prepare_dataset(tokenizer, split="train", max_length=256, limit=None, num_proc=None, batch_size=1000):
    """
    MNLI 데이터셋을 로드하고 토크나이즈합니다.
    
    Args:
        tokenizer: 토크나이저
        split: 데이터셋 스플릿 ("train", "validation_matched", "validation_mismatched")
        max_length: 최대 시퀀스 길이
        limit: 데이터 제한 (디버깅용, None이면 전체)
        num_proc: 멀티프로세싱에 사용할 프로세스 수 (None이면 CPU 코어 수 사용)
        batch_size: 배치 토크나이징에 사용할 배치 크기 (기본값: 1000)
    
    Returns:
        tokenized: 토크나이즈된 데이터셋
    """
    # MNLI 데이터 로드
    data = load_mnli_raw(split=split, limit=limit)
    
    # 배치 토크나이즈 (멀티프로세싱 지원)
    tokenized = data.map(
        lambda ex: tokenize_fn_batch(ex, tokenizer, max_length=max_length),
        batched=True,
        batch_size=batch_size,
        remove_columns=data.column_names,
        desc=f"Tokenizing {split}",
        num_proc=num_proc,
    )
    
    # label이 없는 예제 필터링
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 0)
    
    return tokenized


def main():
    # 1) Load MNLI
    ds = load_dataset("glue", "mnli")
    train = ds["train"]

    # 2) Qwen tokenizer (change to the exact checkpoint you use)
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Ensure pad token exists (important for padding in DataLoader)
    if tokenizer.pad_token is None:
        # Common safe fallback for decoder LMs
        tokenizer.pad_token = tokenizer.eos_token

    # 3) Tokenize with HF map
    train_tok = train.map(
        lambda ex: tokenize_fn(ex, tokenizer, max_length=256),
        remove_columns=train.column_names,
    )

    # 4) Torch DataLoader
    loader = DataLoader(
        train_tok,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    # 5) Inspect one batch
    batch = next(iter(loader))
    print("Batch keys:", batch.keys())
    print("input_ids:", batch["input_ids"].shape)
    print("attention_mask:", batch["attention_mask"].shape)
    print("labels:", batch["labels"].shape)

    print("\n--- example prompt (batch[0]) ---")
    print(batch["input_text"][0])
    print("--- target (batch[0]) ---")
    print(batch["target_text"][0])

    # Optional: check masking looks reasonable
    # Count non-masked label tokens
    non_mask = (batch["labels"][0] != -100).sum().item()
    print("\nnon-masked label tokens in sample 0:", non_mask)

if __name__ == "__main__":
    main()
