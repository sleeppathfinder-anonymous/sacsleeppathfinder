# -*- coding: utf-8 -*-
# Converted from Jupyter Notebook on 2025-10-18T03:57:57
# Markdown cells are preserved as comments.


# ==== Cell 1 (code) ====
from pynvml import nvmlInit, <redacted_token>, nvmlDeviceGetMemoryInfo

nvmlInit()
handle = <redacted_token>(0)
info = nvmlDeviceGetMemoryInfo(handle)
print(f"💾 GPU Memory - Used: {info.used / 1024 ** 3:.2f} GB / Total: {info.total / 1024 ** 3:.2f} GB")

# ==== Cell 2 (code) ====
import pynvml

pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()

for i in range(device_count):
    handle = pynvml.<redacted_token>(i)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU {i} memory used: {mem_info.used / 1024**2} MB")

# ==== Cell 3 (code) ====
import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# ==== Cell 4 (code) ====
from transformers import AutoConfig
config = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
print(config.max_position_embeddings)

# ==== Cell 5 (code) ====
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# 경로 설정
data_dir = "../data"
train_files = [
    f"{data_dir}/train_chunk_I.jsonl",
    f"{data_dir}/train_chunk_II.jsonl",
    f"{data_dir}/train_chunk_III.jsonl"
]

# LLaMA-3 tokenizer 로딩
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# 토큰 길이 측정
all_lengths = []
for file in train_files:
    dataset = load_dataset("json", data_files={"train": file})["train"]
    for example in dataset:
        prompt = f"{example['instruction']}\nInput: {example['input']}\nOutput:"
        input_ids = tokenizer(prompt, truncation=False)["input_ids"]
        all_lengths.append(len(input_ids))

# 통계 출력
print(f"📏 Max length: {max(all_lengths)}")
print(f"📊 Mean length: {np.mean(all_lengths):.2f}")
print(f"🔟 90th percentile: {np.percentile(all_lengths, 90):.0f}")
print(f"🔟 95th percentile: {np.percentile(all_lengths, 95):.0f}")

# ==== Cell 6 (code) ====
import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq, BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType, <redacted_token>
from dotenv import load_dotenv

# .env 및 토큰 로드
load_dotenv("../.env")
HF_ACCESS_TOKEN = os.getenv('SECRET')
if HF_ACCESS_TOKEN is None:
    raise ValueError("HF_ACCESS_TOKEN missing!")

# Config
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
data_dir = "../data"
train_files = [
    f"{data_dir}/train_chunk_I.jsonl",
    f"{data_dir}/train_chunk_II.jsonl",
    f"{data_dir}/train_chunk_III.jsonl"
]
eval_file = f"{data_dir}/valid.jsonl"
output_dir = "../model"
max_seq_length = 256  # 필요에 따라 조정 가능

# 데이터셋 로드
dataset = load_dataset("json", data_files={"validation": eval_file})
train_datasets = [load_dataset("json", data_files={"train": f})["train"] for f in train_files]
dataset["train"] = concatenate_datasets(train_datasets)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_ACCESS_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# 배치 처리 전처리 함수: 각 예제에 대해 prompt 구성 후 한 번에 토큰화
def preprocess(examples):
    prompts = [
        f"{instr}\nInput: {inp}\nOutput: {out}"
        for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
    ]
    tokenized = tokenizer(
        prompts,
        max_length=max_seq_length,
        truncation=True,
        padding="max_length"
    )
    # labels는 input_ids와 동일하게 설정
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(
    preprocess,
    remove_columns=dataset["train"].column_names,
    batched=True
)

# BitsAndBytesConfig를 사용하여 8비트 로딩 및 CPU offload 옵션 설정
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    <redacted_token>=True
)

# 모델 로드 (quantization_config와 device_map 사용)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_auth_token=HF_ACCESS_TOKEN,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 8비트 학습에 적합하도록 모델 준비
model = <redacted_token>(model, 8)

# LoRA 설정 (기본 설정 그대로 사용)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, lora_config)

# Collator 설정
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    <redacted_token>=4,
    <redacted_token>=4,
    <redacted_token>=8,
    warmup_steps=5,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    logging_steps=10,
    logging_strategy="steps",
    save_strategy="no",
    report_to="none"
)

# Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 학습 및 모델 저장
trainer.train()
trainer.save_model(output_dir)

# ==== Cell 7 (code) ====
import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq, BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType, <redacted_token>
from dotenv import load_dotenv

# .env 및 토큰 로드
load_dotenv("../.env")
HF_ACCESS_TOKEN = os.getenv('SECRET')
if HF_ACCESS_TOKEN is None:
    raise ValueError("HF_ACCESS_TOKEN missing!")

# Config
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
data_dir = "../data"
train_files = [
    f"{data_dir}/train_chunk_I.jsonl",
    f"{data_dir}/train_chunk_II.jsonl",
    f"{data_dir}/train_chunk_III.jsonl"
]
eval_file = f"{data_dir}/valid.jsonl"
output_dir = "../model"
max_seq_length = 256  # 필요에 따라 조정 가능

# 데이터셋 로드
dataset = load_dataset("json", data_files={"validation": eval_file})
train_datasets = [load_dataset("json", data_files={"train": f})["train"] for f in train_files]
dataset["train"] = concatenate_datasets(train_datasets)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_ACCESS_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# 배치 처리 전처리 함수: 각 예제에 대해 prompt 구성 후 한 번에 토큰화
def preprocess(examples):
    prompts = [
        f"{instr}\nInput: {inp}\nOutput: {out}"
        for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
    ]
    tokenized = tokenizer(
        prompts,
        max_length=max_seq_length,
        truncation=True,
        padding="max_length"
    )
    # labels는 input_ids와 동일하게 설정
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(
    preprocess,
    remove_columns=dataset["train"].column_names,
    batched=True
)

# BitsAndBytesConfig를 사용하여 8비트 로딩 및 CPU offload 옵션 설정
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    <redacted_token>=True
)

# 모델 로드 (quantization_config와 device_map 사용)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_auth_token=HF_ACCESS_TOKEN,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 8비트 학습에 적합하도록 모델 준비
model = <redacted_token>(model, 8)

# LoRA 설정 (기본 설정 그대로 사용)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, lora_config)

# Collator 설정
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    <redacted_token>=4,
    <redacted_token>=4,
    <redacted_token>=8,
    warmup_steps=5,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    logging_steps=10,
    logging_strategy="steps",
    save_strategy="steps",       # 스텝 단위로 저장
    save_steps=500,              # 예를 들어 500 스텝마다 저장
    save_total_limit=2,          # 최신 체크포인트 2개만 유지
    report_to="none"
)

# Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 학습 및 모델 저장
trainer.train()
trainer.save_model(output_dir)

# ==== Cell 8 (code) ====
results = trainer.evaluate()
print(results)

# ==== Cell 9 (code) ====
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from dotenv import load_dotenv

# 모델 이동을 하지 않도록 하기 위해 사용자 정의 Trainer 클래스 정의
class NoMoveTrainer(Trainer):
    def _move_model_to_device(self, model, device):
        # 이미 모델이 올바른 device (CPU)로 로드되어 있으므로 아무 작업도 하지 않음
        return model

# .env 및 토큰 로드
load_dotenv("../.env")
HF_ACCESS_TOKEN = os.getenv('SECRET')
if HF_ACCESS_TOKEN is None:
    raise ValueError("HF_ACCESS_TOKEN missing!")

# 설정
output_dir = "../model"        # 학습 후 저장한 모델 디렉터리
data_dir = "../data"
eval_file = f"{data_dir}/valid.jsonl"
max_seq_length = 256           # 학습 시 사용했던 값과 동일하게

# 토크나이저 로드 (학습 시 사용했던 모델과 동일하게)
tokenizer = AutoTokenizer.from_pretrained(output_dir)
tokenizer.pad_token = tokenizer.eos_token

# 전처리 함수 (배치 처리)
def preprocess(examples):
    prompts = [
        f"{instr}\nInput: {inp}\nOutput: {out}"
        for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
    ]
    tokenized = tokenizer(
        prompts,
        max_length=max_seq_length,
        truncation=True,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# 평가 데이터셋 로드 및 전처리
dataset = load_dataset("json", data_files={"validation": eval_file})
tokenized_dataset = dataset.map(
    preprocess,
    remove_columns=dataset["validation"].column_names,
    batched=True
)
eval_dataset = tokenized_dataset["validation"]

# DataCollator 생성 (모델은 None으로 설정해도 무방)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None, padding=True)

# 저장된 모델을 CPU 전용으로 불러오기
# device_map={"": "cpu"}와 low_cpu_mem_usage=True를 사용하여 모델의 모든 모듈을 CPU에 로드합니다.
model = AutoModelForCausalLM.from_pretrained(
    output_dir,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True,
    use_auth_token=HF_ACCESS_TOKEN
)

# 평가용 TrainingArguments 설정 (GPU 없이 평가)
eval_args = TrainingArguments(
    output_dir="./eval_output",
    <redacted_token>=4,
    no_cuda=True,         # CPU 전용 평가
    report_to="none"
)

# NoMoveTrainer를 사용하여 Trainer 생성 (모델 이동 관련 오류 회피)
trainer = NoMoveTrainer(
    model=model,
    args=eval_args,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 평가 실행
results = trainer.evaluate()
print("Evaluation results:", results)

# ==== Cell 10 (code) ====
import json

# ==== Cell 11 (code) ====
# 결과를 JSON 파일로 저장
with open("eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)