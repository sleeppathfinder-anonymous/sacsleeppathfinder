# -*- coding: utf-8 -*-
# Converted from Jupyter Notebook on 2025-10-18T03:57:57
# Markdown cells are preserved as comments.


# ==== Cell 1 (code) ====
import pandas as pd
import torch
from datasets import Dataset
from transformers import RobertaTokenizer, <redacted_token>, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.utils import resample

# ==================== 1. 데이터 로드 및 전처리 ====================
def <redacted_token>(file_path):
    """CSV 파일을 로드하고 결측치를 제거합니다."""
    df = pd.read_csv(file_path)
    df.dropna(subset=['User Utterance', 'Intent'], inplace=True)
    return df

# ==================== 2. 데이터 불균형 처리 (오버샘플링) ====================
def oversample_data(df):
    """
    라벨 불균형을 해결하기 위해 소수 클래스를 오버샘플링합니다.
    가장 많은 라벨의 개수에 맞춰 다른 라벨의 데이터 수를 늘립니다.
    """
    max_count = df['Intent'].value_counts().max()
    oversampled_df = pd.DataFrame()
    for intent in df['Intent'].unique():
        intent_df = df[df['Intent'] == intent]
        oversampled_intent_df = resample(intent_df, replace=True, n_samples=max_count, random_state=42)
        oversampled_df = pd.concat([oversampled_df, oversampled_intent_df])
    return oversampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

# ==================== 3. 모델 학습 및 평가 지표 정의 ====================
def compute_metrics(eval_pred):
    """모델의 성능을 평가하기 위한 지표(정확도, F1-macro)를 계산합니다."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro
    }

# ==================== 4. RoBERta 모델 재학습 코드 ====================
def train_roberta_model(train_df, val_df, model_name="roberta-base"):
    """RoBERTa 모델을 로드하고 미세조정(fine-tuning)합니다."""
    # 라벨을 숫자 ID로 매핑
    unique_labels = sorted(train_df['Intent'].unique())
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    num_labels = len(unique_labels)
    
    train_df['labels'] = train_df['Intent'].map(label_to_id)
    val_df['labels'] = val_df['Intent'].map(label_to_id)

    # 데이터셋 생성
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # 토크나이저 및 모델 로드
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = <redacted_token>.from_pretrained(model_name, num_labels=num_labels)

    # 데이터 토큰화 함수
    def tokenize_function(examples):
        return tokenizer(examples['User Utterance'], padding="max_length", truncation=True)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # 학습에 불필요한 컬럼 제거
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(["User Utterance", "Intent", "Original Selftext"])
    tokenized_val_dataset = tokenized_val_dataset.remove_columns(["User Utterance", "Intent", "Original Selftext"])

    # Hugging Face Trainer를 위한 학습 인자 설정
    training_args = TrainingArguments(
        output_dir='./<redacted_token>',
        num_train_epochs=3,
        <redacted_token>=8,
        <redacted_token>=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./roberta_logs',
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",                # ✅ 여기! (evaluation_strategy -> eval_strategy)
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",                    # 문제가 되면 이 줄은 삭제하거나 report_to=[] 로 바꾸세요
    )

    # Trainer 초기화 및 학습
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        # evaluation_strategy="epoch",        # ❌ 삭제: TrainingArguments에서만 설정
    )

    trainer.train()
    
    # 학습된 모델과 토크나이저 저장
    output_dir = "./roberta_finetuned"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ 모델 학습 완료! 학습된 모델이 '{output_dir}'에 저장되었습니다.")
    
# ==================== 5. 실행부 (수정) ====================
if __name__ == "__main__":
    file_path = "../data/<redacted_token>.csv"
    df = <redacted_token>(file_path)

    # 1. 원본 데이터를 먼저 학습/검증 데이터로 분할
    train_df_original, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Intent'])
    
    # 2. 오직 학습 데이터에만 오버샘플링 적용
    print("🚀 학습 데이터셋의 불균형을 해결하기 위해 오버샘플링을 시작합니다...")
    train_df_oversampled = oversample_data(train_df_original)
    print("✅ 오버샘플링 완료. 학습 데이터셋의 모든 라벨이 동일한 데이터 수를 가집니다.")
    print(train_df_oversampled['Intent'].value_counts())
    
    print("\n✅ 데이터 분할 및 오버샘플링 완료. 모델 재학습을 시작합니다.")
    
    # 3. 오버샘플링된 학습 데이터와 원본 검증 데이터를 사용하여 모델 학습
    train_roberta_model(train_df_oversampled, val_df)

# ==== Cell 2 (code) ====
import pandas as pd
import torch
from datasets import Dataset
from transformers import RobertaTokenizer, <redacted_token>, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.utils import resample

# ==================== 1. 데이터 로드 및 전처리 ====================
def <redacted_token>(file_path):
    """CSV 파일을 로드하고 결측치를 제거합니다."""
    df = pd.read_csv(file_path)
    df.dropna(subset=['User Utterance', 'Intent'], inplace=True)
    return df

# ==================== 2. 데이터 불균형 처리 (언더샘플링) ====================
def undersample_data(df):
    """
    라벨 불균형을 해결하기 위해 다수 클래스를 언더샘플링합니다.
    가장 적은 라벨의 개수에 맞춰 다른 라벨의 데이터 수를 줄입니다.
    """
    min_count = df['Intent'].value_counts().min()
    undersampled_df = pd.DataFrame()
    for intent in df['Intent'].unique():
        intent_df = df[df['Intent'] == intent]
        undersampled_intent_df = resample(intent_df, replace=False, n_samples=min_count, random_state=42)
        undersampled_df = pd.concat([undersampled_df, undersampled_intent_df])
    return undersampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

# ==================== 3. 모델 학습 및 평가 지표 정의 ====================
def compute_metrics(eval_pred):
    """모델의 성능을 평가하기 위한 지표(정확도, F1-macro)를 계산합니다."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro
    }

# ==================== 4. RoBERta 모델 재학습 코드 ====================
def train_roberta_model(train_df, val_df, model_name="roberta-base"):
    """RoBERTa 모델을 로드하고 미세조정(fine-tuning)합니다."""
    # 라벨을 숫자 ID로 매핑
    unique_labels = sorted(train_df['Intent'].unique())
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    num_labels = len(unique_labels)
    
    train_df['labels'] = train_df['Intent'].map(label_to_id)
    val_df['labels'] = val_df['Intent'].map(label_to_id)

    # 데이터셋 생성
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # 토크나이저 및 모델 로드
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = <redacted_token>.from_pretrained(model_name, num_labels=num_labels)

    # 데이터 토큰화 함수
    def tokenize_function(examples):
        return tokenizer(examples['User Utterance'], padding="max_length", truncation=True)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # 학습에 불필요한 컬럼 제거
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(["User Utterance", "Intent", "Original Selftext"])
    tokenized_val_dataset = tokenized_val_dataset.remove_columns(["User Utterance", "Intent", "Original Selftext"])

    # Hugging Face Trainer를 위한 학습 인자 설정
    training_args = TrainingArguments(
        output_dir='./<redacted_token>',
        num_train_epochs=3,
        <redacted_token>=8,
        <redacted_token>=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./roberta_logs',
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",                # ✅ 여기! (evaluation_strategy -> eval_strategy)
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",                    # 문제가 되면 이 줄은 삭제하거나 report_to=[] 로 바꾸세요
    )

    # Trainer 초기화 및 학습
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        # evaluation_strategy="epoch",        # ❌ 삭제: TrainingArguments에서만 설정
    )

    trainer.train()
    
    # 학습된 모델과 토크나이저 저장
    output_dir = "./<redacted_token>"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ 모델 학습 완료! 학습된 모델이 '{output_dir}'에 저장되었습니다.")


# ==================== 5. 실행부 (수정) ====================
if __name__ == "__main__":
    file_path = "../data/<redacted_token>.csv"
    df = <redacted_token>(file_path)

    # 1. 원본 데이터를 먼저 학습/검증 데이터로 분할
    train_df_original, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Intent'])
    
    # 2. 오직 학습 데이터에만 언더샘플링 적용
    print("🚀 학습 데이터셋의 불균형을 해결하기 위해 언더샘플링을 시작합니다...")
    train_df_undersampled = undersample_data(train_df_original)
    print("✅ 언더샘플링 완료. 학습 데이터셋의 모든 라벨이 동일한 데이터 수를 가집니다.")
    print(train_df_undersampled['Intent'].value_counts())
    
    print("\n✅ 데이터 분할 및 언더샘플링 완료. 모델 재학습을 시작합니다.")
    
    # 3. 언더샘플링된 학습 데이터와 원본 검증 데이터를 사용하여 모델 학습
    train_roberta_model(train_df_undersampled, val_df)

# ==== Cell 3 (code) ====
import pandas as pd
import torch
from datasets import Dataset
from transformers import RobertaTokenizer, <redacted_token>, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.utils import resample

# ==================== 1. 데이터 로드 및 전처리 ====================
def <redacted_token>(file_path):
    """CSV 파일을 로드하고 결측치를 제거합니다."""
    df = pd.read_csv(file_path)
    df.dropna(subset=['User Utterance', 'Intent'], inplace=True)
    return df

# ==================== 2. 데이터 불균형 처리 (언더샘플링) ====================
def undersample_data(df):
    """
    라벨 불균형을 해결하기 위해 다수 클래스를 언더샘플링합니다.
    가장 적은 라벨의 개수에 맞춰 다른 라벨의 데이터 수를 줄입니다.
    """
    min_count = df['Intent'].value_counts().min()
    undersampled_df = pd.DataFrame()
    for intent in df['Intent'].unique():
        intent_df = df[df['Intent'] == intent]
        undersampled_intent_df = resample(intent_df, replace=False, n_samples=min_count, random_state=42)
        undersampled_df = pd.concat([undersampled_df, undersampled_intent_df])
    return undersampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

# ==================== 3. 모델 학습 및 평가 지표 정의 ====================
def compute_metrics(eval_pred):
    """모델의 성능을 평가하기 위한 지표(정확도, F1-macro)를 계산합니다."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro
    }

# ==================== 4. RoBERta 모델 재학습 코드 ====================
def train_roberta_model(train_df, val_df, model_name="roberta-base"):
    """RoBERTa 모델을 로드하고 미세조정(fine-tuning)합니다."""
    # 라벨을 숫자 ID로 매핑
    unique_labels = sorted(train_df['Intent'].unique())
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    num_labels = len(unique_labels)
    
    train_df['labels'] = train_df['Intent'].map(label_to_id)
    val_df['labels'] = val_df['Intent'].map(label_to_id)

    # 데이터셋 생성
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # 토크나이저 및 모델 로드
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = <redacted_token>.from_pretrained(model_name, num_labels=num_labels)

    # 데이터 토큰화 함수
    def tokenize_function(examples):
        return tokenizer(examples['User Utterance'], padding="max_length", truncation=True)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # 학습에 불필요한 컬럼 제거
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(["User Utterance", "Intent", "Original Selftext"])
    tokenized_val_dataset = tokenized_val_dataset.remove_columns(["User Utterance", "Intent", "Original Selftext"])

    # Hugging Face Trainer를 위한 학습 인자 설정
    training_args = TrainingArguments(
        output_dir='./<redacted_token>',
        num_train_epochs=3,
        <redacted_token>=8,
        <redacted_token>=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./roberta_logs',
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",                # ✅ 여기! (evaluation_strategy -> eval_strategy)
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",                    # 문제가 되면 이 줄은 삭제하거나 report_to=[] 로 바꾸세요
    )

    # Trainer 초기화 및 학습
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        # evaluation_strategy="epoch",        # ❌ 삭제: TrainingArguments에서만 설정
    )

    trainer.train()
    
    # 학습된 모델과 토크나이저 저장
    output_dir = "./<redacted_token>"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ 모델 학습 완료! 학습된 모델이 '{output_dir}'에 저장되었습니다.")


# ==================== 5. 실행부 (수정) ====================
if __name__ == "__main__":
    file_path = "../data/<redacted_token>.csv"
    df = <redacted_token>(file_path)

    # 1. 원본 데이터를 먼저 학습/검증 데이터로 분할
    train_df_original, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Intent'])
    
    # 2. 오직 학습 데이터에만 언더샘플링 적용
    print("🚀 학습 데이터셋의 불균형을 해결하기 위해 언더샘플링을 시작합니다...")
    train_df_undersampled = undersample_data(train_df_original)
    print("✅ 언더샘플링 완료. 학습 데이터셋의 모든 라벨이 동일한 데이터 수를 가집니다.")
    print(train_df_undersampled['Intent'].value_counts())
    
    print("\n✅ 데이터 분할 및 언더샘플링 완료. 모델 재학습을 시작합니다.")
    
    # 3. 언더샘플링된 학습 데이터와 원본 검증 데이터를 사용하여 모델 학습
    train_roberta_model(train_df_undersampled, val_df)

# ==== Cell 4 (code) ====
import pandas as pd
import torch
from datasets import Dataset
from transformers import RobertaTokenizer, <redacted_token>, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.utils import resample
import os

# ==================== 1. 데이터 로드 및 전처리 ====================
def <redacted_token>(file_path):
    """CSV 파일을 로드하고 결측치를 제거합니다."""
    df = pd.read_csv(file_path)
    df.dropna(subset=['User Utterance', 'Intent'], inplace=True)
    return df

# ==================== 2. 데이터 불균형 처리 (언더샘플링) ====================
def undersample_data(df):
    """
    라벨 불균형을 해결하기 위해 다수 클래스를 언더샘플링합니다.
    가장 적은 라벨의 개수에 맞춰 다른 라벨의 데이터 수를 줄입니다.
    """
    min_count = df['Intent'].value_counts().min()
    undersampled_df = pd.DataFrame()
    for intent in df['Intent'].unique():
        intent_df = df[df['Intent'] == intent]
        undersampled_intent_df = resample(intent_df, replace=False, n_samples=min_count, random_state=42)
        undersampled_df = pd.concat([undersampled_df, undersampled_intent_df])
    return undersampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

# ==================== 3. 모델 학습 및 평가 지표 정의 ====================
def compute_metrics(eval_pred):
    """모델의 성능을 평가하기 위한 지표(정확도, F1-macro)를 계산합니다."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro
    }

# ==================== 4. RoBERta 모델 재학습 코드 (K-Fold 적용) ====================
def <redacted_token>(df, model_name="roberta-base", n_splits=5):
    """
    RoBERTa 모델을 K-Fold 교차 검증을 사용하여 미세조정(fine-tuning)하고,
    언더샘플링을 적용합니다.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    
    # K-Fold 반복
    for fold, (train_index, val_index) in enumerate(kf.split(df)):
        print(f"==================== 🚀 Fold {fold+1}/{n_splits} 시작 ====================")
        
        # 원본 데이터를 학습/검증 데이터로 분할
        train_df_original = df.iloc[train_index]
        val_df = df.iloc[val_index]
        
        # 오직 학습 데이터에만 언더샘플링 적용
        train_df_undersampled = undersample_data(train_df_original)
        
        print(f"✅ Fold {fold+1}: 언더샘플링된 학습 데이터 수: {len(train_df_undersampled)}, 검증 데이터 수: {len(val_df)}")
        print(f"✅ 학습 데이터 라벨 분포:\n{train_df_undersampled['Intent'].value_counts()}")

        # 라벨을 숫자 ID로 매핑
        unique_labels = sorted(train_df_undersampled['Intent'].unique())
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        num_labels = len(unique_labels)
        
        train_df_undersampled['labels'] = train_df_undersampled['Intent'].map(label_to_id)
        val_df['labels'] = val_df['Intent'].map(label_to_id)

        # Hugging Face Dataset 객체로 변환
        train_dataset = Dataset.from_pandas(train_df_undersampled)
        val_dataset = Dataset.from_pandas(val_df)
        
        # 토크나이저 및 모델 로드
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = <redacted_token>.from_pretrained(model_name, num_labels=num_labels)

        def tokenize_function(examples):
            return tokenizer(examples['User Utterance'], padding="max_length", truncation=True)

        tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
        tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        # 학습에 불필요한 컬럼 제거
        tokenized_train_dataset = tokenized_train_dataset.remove_columns(["User Utterance", "Intent", "Original Selftext"])
        tokenized_val_dataset = tokenized_val_dataset.remove_columns(["User Utterance", "Intent", "Original Selftext"])

        # 학습 인자 설정
        output_dir = f'./<redacted_token>/fold_{fold+1}'
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=10,  # Early Stopping을 위해 충분한 epoch 설정
            <redacted_token>=8,
            <redacted_token>=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./roberta_logs',
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro", # 성능 평가 지표를 F1-macro로 변경
            greater_is_better=True,
            report_to="none",
            save_total_limit=1 # 최적 모델만 저장
        )

        # Trainer 초기화 및 학습 (EarlyStoppingCallback 추가)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # patience 설정
        )

        trainer.train()
        
        # 폴드별 성능 평가
        eval_results = trainer.evaluate()
        fold_results.append(eval_results)
        print(f"✅ Fold {fold+1} 결과: {eval_results}")
        
        # 최적 모델 저장
        best_model_dir = f'./<redacted_token>/fold_{fold+1}'
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
            
        trainer.model.save_pretrained(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)
        print(f"✅ Fold {fold+1}의 최적 모델이 '{best_model_dir}'에 저장되었습니다.")
        
    # 모든 폴드 결과 요약
    print("\n==================== ✅ K-Fold 학습 완료 ====================")
    avg_accuracy = np.mean([res['eval_accuracy'] for res in fold_results])
    avg_f1_macro = np.mean([res['eval_f1_macro'] for res in fold_results])
    
    print(f"⭐ 전체 폴드 평균 정확도: {avg_accuracy:.4f}")
    print(f"⭐ 전체 폴드 평균 F1-macro: {avg_f1_macro:.4f}")

# ==================== 5. 실행부 ====================
if __name__ == "__main__":
    file_path = "../data/<redacted_token>.csv"
    df = <redacted_token>(file_path)
    
    print("🚀 K-Fold 교차 검증을 사용하여 모델 학습을 시작합니다.")
    <redacted_token>(df, n_splits=5)

# ==== Cell 5 (code) ====