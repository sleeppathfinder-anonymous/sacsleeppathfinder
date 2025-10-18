# pilot_chatbot.py (ANONYMIZED)
import os
import uuid
import gradio as gr
import torch
import pandas as pd
import re
import json
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline,
)
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sentence_transformers import SentenceTransformer
import faiss
import warnings
import openai
import time
import random
from peft import PeftConfig, get_peft_model

# =========================
# 0. Safe Defaults & Flags
# =========================
warnings.filterwarnings("ignore")

DEBUG = os.getenv("DEBUG", "false").lower() == "true"  # default False
SAVE_HISTORY = os.getenv("SAVE_HISTORY", "false").lower() == "true"
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"

# =========================
# 1. API Keys (ENV ONLY)
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment variables.")
openai.api_key = OPENAI_API_KEY  # legacy compat

HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
if not HF_ACCESS_TOKEN:
    raise ValueError("Missing HF_ACCESS_TOKEN in environment variables.")

# =========================
# 2. Model IDs (Anonymized)
# =========================
ROBERTA_MODEL_NAME = os.getenv(
    "ROBERTA_MODEL_NAME", "orgname/roberta-cbti-finetuned-anon"
)
LLAMA_BASE_MODEL = os.getenv(
    "LLAMA_BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct"
)
LLAMA_LORA_ADAPTER = os.getenv(
    "LLAMA_LORA_ADAPTER", "orgname/cbti-lora-anon"
)

# =========================
# 3. Util Functions
# =========================
def translate_to_korean(text: str) -> str:
    """KR translation; safe: no console echo of the source text."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional Korean translator for mental health chatbot dialogues using CBT-I techniques. "
                        "Translate the following English sentence into natural, emotionally supportive Korean suitable for a CBT-I therapy session. "
                        "Return ONLY the translated Korean sentence."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.4,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "잠시 번역에 문제가 발생했어요. 다시 시도해 주세요."

def translate_to_english(text: str) -> str:
    """EN translation; safe: no console echo of the source text."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates Korean to English."},
                {"role": "user", "content": f"Translate the following Korean sentence to English:\n\n{text}"},
            ],
            temperature=0.3,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Temporary translation issue. Please try again."

def normalize_yes_no(user_input):
    normalized = user_input.strip().lower().replace(" ", "")
    if normalized in ["예", "네", "ㅇ", "y", "yes"]:
        return "예"
    elif normalized in ["아니오", "아니요", "아뇨", "ㄴ", "n", "no"]:
        return "아니오"
    else:
        return None

def postprocess_response(text: str) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if not re.search(r'[.!?]"?$', clean):
        clean += "."
    return clean

# =========================
# 4. Device & Pipelines
# =========================
gpu_device = os.environ.get("GPU_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
llama_device = int(gpu_device.split(":")[1]) if ":" in gpu_device else 0 if gpu_device.startswith("cuda") else -1

# Intent classifier (RoBERTa)
bert_tokenizer = RobertaTokenizer.from_pretrained(
    ROBERTA_MODEL_NAME, use_auth_token=HF_ACCESS_TOKEN
)
bert_model = RobertaForSequenceClassification.from_pretrained(
    ROBERTA_MODEL_NAME, use_auth_token=HF_ACCESS_TOKEN
)
bert_model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

# LLaMA-3.1 + LoRA (adapter merged)
llama_tokenizer = AutoTokenizer.from_pretrained(
    LLAMA_BASE_MODEL,
    use_auth_token=HF_ACCESS_TOKEN,
)

base_llama_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_BASE_MODEL,
    use_auth_token=HF_ACCESS_TOKEN,
    torch_dtype=torch.float32,
    device_map="auto",
)

peft_config = PeftConfig.from_pretrained(
    LLAMA_LORA_ADAPTER,
    use_auth_token=HF_ACCESS_TOKEN,
)
lora_llama_model = get_peft_model(base_llama_model, peft_config)
lora_llama_model.eval()
merged_llama_model = lora_llama_model.merge_and_unload()
merged_llama_model.eval()

llama_pipeline = pipeline(
    "text-generation",
    model=merged_llama_model,
    tokenizer=llama_tokenizer,
    device_map="auto",
)
llama_pipeline.model.eval()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# 5. Data Loading (Safe)
# =========================
RAG_XLSX_PATH = os.getenv("RAG_XLSX_PATH")  # optional

def _build_df_from_excel(path: str) -> pd.DataFrame:
    data_df = pd.read_excel(path)
    return pd.DataFrame(
        [
            {
                "approach": row["Approach"],
                "user_input": row["User Utterance (English)"],
                "info": row["Therapist Response (English)"],
            }
            for _, row in data_df.iterrows()
        ]
    )

def _build_sample_df() -> pd.DataFrame:
    # Minimal, non-sensitive stub data for anonymous reproduction
    rows = [
        {
            "approach": "Sleep Restriction",
            "user_input": "I stay in bed for hours but only sleep a few.",
            "info": "Try limiting your time in bed to your actual sleep time, then gradually increase it to consolidate sleep efficiency.",
        },
        {
            "approach": "Stimulus Control",
            "user_input": "I lie in bed scrolling my phone for a long time.",
            "info": "Reserve the bed for sleep; if you can’t fall asleep in ~20 minutes, get up and do a low-stimulation activity.",
        },
        {
            "approach": "Sleep Hygiene",
            "user_input": "I drink coffee late and use bright screens at night.",
            "info": "Reduce late caffeine and bright screens; keep consistent sleep/wake times and optimize light exposure.",
        },
        {
            "approach": "Relaxation Techniques",
            "user_input": "I feel very tense at bedtime.",
            "info": "Practice slow breathing or progressive muscle relaxation to lower arousal before bed.",
        },
        {
            "approach": "Cognitive Restructuring",
            "user_input": "If I can’t sleep tonight, my day will be ruined.",
            "info": "Examine the evidence for this thought and consider more balanced alternatives.",
        },
    ]
    return pd.DataFrame(rows)

if RAG_XLSX_PATH and os.path.exists(RAG_XLSX_PATH):
    df = _build_df_from_excel(RAG_XLSX_PATH)
else:
    df = _build_sample_df()

CBT_I_DESCRIPTIONS = {
    "수면 제한 요법 (Sleep Restriction)": (
        "수면 제한 요법은 침대에 머무는 시간을 의도적으로 줄여, 침대와 수면 사이의 올바른 연결고리를 재구축하는 방법입니다. "
        "처음에는 실제 수면 시간만큼 침대에서 자고 점차 시간을 늘려가며 ‘침대=숙면’ 연합을 강화합니다."
    ),
    "자극 조절 요법 (Stimulus Control)": (
        "자극 조절 요법은 침대를 오직 수면을 위한 장소로 재정립하는 치료법입니다. 잠이 오지 않으면 즉시 침대에서 벗어나고, 침대에서는 수면 외 활동을 피합니다."
    ),
    "수면 위생 교육 (Sleep Hygiene)": (
        "생활 습관을 개선하는 방법입니다. 예: 늦은 카페인 제한, 일정한 취침·기상 시간 유지, 취침 전 밝은 화면·강한 조명 줄이기 등."
    ),
    "이완 요법 (Relaxation Techniques)": (
        "심리적·신체적 긴장을 낮춰 자연스러운 수면을 돕습니다. 심호흡, 점진적 근육 이완, 가벼운 명상 등."
    ),
    "인지적 재구성 (Cognitive Restructuring)": (
        "수면 관련 부정적 자동사고를 균형 잡힌 생각으로 재구성하여 불안을 낮추는 기법입니다."
    ),
}

ENG_TO_KOR_KEY = {
    "Sleep Restriction": "수면 제한 요법 (Sleep Restriction)",
    "Stimulus Control": "자극 조절 요법 (Stimulus Control)",
    "Sleep Hygiene": "수면 위생 교육 (Sleep Hygiene)",
    "Relaxation Techniques": "이완 요법 (Relaxation Techniques)",
    "Cognitive Restructuring": "인지적 재구성 (Cognitive Restructuring)",
}
KOR_TO_ENG_KEY = {v: k for k, v in ENG_TO_KOR_KEY.items()}

# =========================
# 6. Session State Helpers
# =========================
def get_initial_state():
    return {
        "session_id": uuid.uuid4().hex,  # no username in filenames/logs
        "nickname": None,  # optional; never stored to filenames
        "history": [],
        "prev_approaches": [],
        "recommended_approach": None,
        "consult_query": None,
        "faiss_index_cache": {},
        "mode": "학습 모드",
        "consulting_active": False,
        "socratic_active": False,
        "socratic_depth": 0,
        "max_depth": 5,
        "socratic_hints": [],
        "current_subquestion": None,
        "current_confidence": "low",
        "current_type": None,
        "self_decision_pending": False,
        "technique_selection_pending": False,
        "waiting_end_confirmation": False,
        "awaiting_termination": False,
        "learning_index": 0,
        "iterative_advice_active": False,
        "iterative_context": "",
        "current_iterative_advice": "",
        "type_history": [],
    }

def reset_state(state):
    # carry over only session_id; rotate to a new one for safety
    new_state = get_initial_state()
    return new_state

# =========================
# 7. Core Functions
# =========================
def extract_type_flexible(text):
    match = re.search(r"Type:\s*(\w+)", text)
    if match:
        t = match.group(1).lower()
        valid = [
            "clarity",
            "assumptions",
            "reasons_evidence",
            "implication_consequences",
            "alternate_viewpoints_perspectives",
        ]
        if t in valid:
            return t
    return None

def classify_intent_with_bert(user_input):
    inputs = bert_tokenizer(
        user_input, return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    inputs = {k: v.to(bert_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = bert_model(**inputs).logits
    predicted_idx = torch.argmax(logits, dim=1).item()
    return list(ENG_TO_KOR_KEY.keys())[predicted_idx]

def retrieve_info_by_approach(query, approach, df, state, top_k=3):
    if approach not in state["faiss_index_cache"]:
        filtered_df = df[df["approach"] == approach]
        if filtered_df.empty:
            return "No reference info available."
        input_list = filtered_df["user_input"].dropna().tolist()
        emb = embedding_model.encode(input_list, convert_to_tensor=True)
        index = faiss.IndexFlatL2(emb.shape[1])
        index.add(emb.cpu().numpy())
        state["faiss_index_cache"][approach] = (index, filtered_df)
    index, filtered_df = state["faiss_index_cache"][approach]
    query_vector = embedding_model.encode([query], convert_to_tensor=True).cpu().numpy()
    _, indices = index.search(query_vector, top_k)
    return filtered_df.iloc[indices[0][0]]["info"]

def merge_hints_to_utterance(hints):
    return " ".join([f'For the question "{q}", the answer was "{a}".' for q, a in hints])

def generate_confidence_field(user_input, context, depth):
    conf_prompt = f"""You are a CBT-I assistant.
Evaluate your confidence in understanding the user's situation.
User's statement: "{user_input}"
Context: "{context}"
Depth: {depth}
Respond ONLY with one word: low, middle, or high.
"""
    out = llama_pipeline(conf_prompt, max_new_tokens=10, temperature=0.4, top_p=0.9)[0][
        "generated_text"
    ]
    conf = out.strip().lower()
    if conf not in ["low", "middle", "high"]:
        conf = "low"
    return conf

def generate_type_field(user_input, context, depth, state):
    type_prompt = """
You are a Socratic Question Classifier for CBT-I sleep therapy chatbot.
Below are 5 Socratic Question Types based on Paul & Elder (2019), each with a description and example:
1) clarity
2) assumptions
3) reasons_evidence
4) implication_consequences
5) alternate_viewpoints_perspectives
--------------------------------------------------
Now classify the user’s statement:
User: "{user_input}"
Context: "{context}"
Respond in the following format only:
Type: <one of clarity, assumptions, reasons_evidence, implication_consequences, alternate_viewpoints_perspectives>
""".strip()
    try:
        out = llama_pipeline(type_prompt, max_new_tokens=60, temperature=0.3, top_p=0.8)[
            0
        ]["generated_text"]
        typ = extract_type_flexible(out) or "clarity"
        state.setdefault("type_history", []).append(typ)
        if len(state["type_history"]) > 3:
            state["type_history"].pop(0)
        if state["type_history"].count("clarity") >= 2:
            typ = random.choice(
                [
                    "assumptions",
                    "reasons_evidence",
                    "implication_consequences",
                    "alternate_viewpoints_perspectives",
                ]
            )
        return typ
    except Exception:
        return "clarity"

def generate_acknowledgment(user_input, context, depth):
    ack_prompt = f"""
You are a compassionate CBT-I assistant. 
User statement: "{user_input}"
Conversation context: "{context}"
Depth: {depth}
Instructions:
1. Generate a brief but emotionally nuanced empathetic acknowledgment.
2. Reflect the specific emotional tone.
3. Avoid generic phrases like "I understand."
Respond ONLY with the empathetic sentence in English.
"""
    out = llama_pipeline(ack_prompt, max_new_tokens=60, temperature=0.5, top_p=0.9)[0][
        "generated_text"
    ]
    ack = out.strip()
    return translate_to_korean(ack)

def generate_subquestion_field(user_input, context, depth, typ):
    prompt = llama_pipeline.tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": (
                    "You are a Socratic therapist helping a user struggling with sleep problems. "
                    "Generate ONE supportive, concrete Socratic question matching the given type."
                ),
            },
            {
                "role": "user",
                "content": f"""
User's concern: "{user_input}"
Previous conversation context: "{context}"
Socratic question type: {typ}
Instructions:
- Only ONE question in English.
- Be specific and supportive.
- No quotes, no explanations.
""",
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    try:
        result = llama_pipeline(
            prompt, max_new_tokens=60, temperature=0.4, top_p=0.9
        )[0]["generated_text"]
        question = result[len(prompt) :].strip()
        if "?" not in question or len(question) < 5:
            fallback = {
                "clarity": "Could you tell me more about what that feels like?",
                "assumptions": "What might you be assuming when that thought comes up?",
                "reasons_evidence": "What makes you think that will happen?",
                "implication_consequences": "What do you think might happen if this continues?",
                "alternate_viewpoints_perspectives": "Is there another way you could view this situation?",
            }
            question = fallback.get(typ, "Could you explain a bit more about that?")
        return question
    except Exception:
        return "Could you explain a bit more about that?"

def generate_full_subquestion_v2(user_input, context="", depth=0, state=None):
    conf = generate_confidence_field(user_input, context, depth)
    typ = generate_type_field(user_input, context, depth, state)
    subq = generate_subquestion_field(user_input, context, depth, typ)
    subq_kr = translate_to_korean(subq)
    ack = generate_acknowledgment(user_input, context, depth)
    state["current_confidence"] = conf
    state["current_type"] = typ
    final_output = f"{ack.strip()} 혹시 {subq_kr.strip().rstrip('.').rstrip('?')}?"
    return postprocess_response(final_output)

def generate_response(user_input, approach_en, context, include_termination=True):
    kor_key = ENG_TO_KOR_KEY.get(approach_en)
    desc_kr = CBT_I_DESCRIPTIONS.get(kor_key, "")
    sentences = re.split(r"(?<=[.!?])\s+", context)
    summary = " ".join(sentences[:2])
    prompt = llama_pipeline.tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": (
                    "You are a warm and empathetic CBT-I therapist. Respond with three clear and concise sentences in Korean: "
                    "(1) empathy, (2) CBT advice for the selected technique, and (3) ask if the user wants to end—only if include_termination is true."
                ),
            },
            {
                "role": "user",
                "content": f"""
User concern: {user_input}
Recommended CBT-I technique: {approach_en}
CBT-I Description (KR): {desc_kr}
Extra context: {summary}
If include_termination is True, end with:
'이 대화를 마치고 싶으신가요? '예' 또는 '아니오'로 대답해 주세요.'
Otherwise, stop after the CBT advice.
""",
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    out = llama_pipeline(prompt, max_new_tokens=180, temperature=0.4, top_p=0.8)[0][
        "generated_text"
    ]
    english_response = re.sub(r"\s+", " ", out[len(prompt) :].strip())
    korean_response = translate_to_korean(english_response)
    return postprocess_response(korean_response)

def generate_self_decision_message(state):
    combined = state["consult_query"] + " " + merge_hints_to_utterance(state["socratic_hints"])
    approach_en = classify_intent_with_bert(combined).strip()
    state["recommended_approach"] = approach_en
    kor_key = ENG_TO_KOR_KEY.get(approach_en)
    desc_kr = CBT_I_DESCRIPTIONS.get(kor_key, "")
    prompt = llama_pipeline.tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "You are a CBT-I expert who provides natural, empathetic recommendations based on user concerns.",
            },
            {
                "role": "user",
                "content": f"""
Concern: {combined}
Recommended CBT-I technique: "{kor_key}" ({approach_en})
Technique (KR): "{desc_kr}"
- Begin with empathy
- Explain why this technique could be helpful
- End with: '이 방법을 시도해보시겠어요? '예' 또는 '아니오'로 답해주세요.'
Return in Korean.
""",
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    raw = llama_pipeline(prompt, max_new_tokens=180, temperature=0.4, top_p=0.8)[0][
        "generated_text"
    ]
    english_reply = re.sub(r"\s+", " ", raw[len(prompt) :].strip())
    korean_reply = translate_to_korean(english_reply)
    termination_kor = "이 방법을 시도해보시겠어요? '예' 또는 '아니오'로 답해주세요."
    if not korean_reply.strip().endswith(termination_kor):
        korean_reply += " " + termination_kor
    return postprocess_response(korean_reply)

def generate_personalized_advice(user_input, last_advice, technique_name):
    prompt = llama_pipeline.tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": (
                    "You are a CBT-I sleep therapist. The user has already received advice about a specific CBT-I technique "
                    "and is now asking a follow-up. Respond warmly in Korean with a personalized suggestion. Avoid repeating full explanations."
                ),
            },
            {
                "role": "user",
                "content": f"""
CBT-I technique: {technique_name}
Previous advice given: "{last_advice}"
User's follow-up message: "{user_input}"
- Respond in Korean
- Be warm and supportive
- Provide a specific suggestion
- Avoid unnecessary questions
""",
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    output = llama_pipeline(prompt, max_new_tokens=180, temperature=0.5, top_p=0.9)[0][
        "generated_text"
    ]
    return postprocess_response(translate_to_korean(output[len(prompt) :].strip()))

def finalize_socratic_and_advice(state):
    merged = merge_hints_to_utterance(state["socratic_hints"])
    query = state["consult_query"] + " " + merged
    approach_en = state["recommended_approach"]
    if not approach_en:
        return "추천 기법이 아직 결정되지 않았습니다."
    info = retrieve_info_by_approach(query, approach_en, df, state)
    korean_response = generate_response(query, approach_en, info, include_termination=False)
    return postprocess_response(
        korean_response + " 추가 의견이 있으시면 입력해 주세요. 만족하시면 '만족'을 입력해 주세요."
    )

def save_history_to_json(state):
    if not SAVE_HISTORY:
        return "설정상 대화 저장이 비활성화되어 있습니다."
    os.makedirs("./history", exist_ok=True)
    filename = f"./history/session_{state['session_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(state["history"], f, ensure_ascii=False, indent=4)
    return f"대화 기록이 저장되었습니다: {filename}"

# =========================
# 8. Mode Handlers
# =========================
TECHNIQUES_ORDER = [
    "수면 제한 요법 (Sleep Restriction)",
    "자극 조절 요법 (Stimulus Control)",
    "수면 위생 교육 (Sleep Hygiene)",
    "이완 요법 (Relaxation Techniques)",
    "인지적 재구성 (Cognitive Restructuring)",
]

def process_learning_mode(user_input, state):
    if user_input.strip() != "":
        idx = state.get("learning_index", 0)
        if idx < len(TECHNIQUES_ORDER):
            technique = TECHNIQUES_ORDER[idx]
            explanation = CBT_I_DESCRIPTIONS.get(technique, "설명이 없습니다.")
            state["learning_index"] = idx + 1
            if state["learning_index"] < len(TECHNIQUES_ORDER):
                msg = (
                    f"[{technique}]\n\n{explanation}\n\n"
                    "다음 기법으로 넘어가기 위해 아무 내용이나 입력해 주세요."
                )
            else:
                msg = (
                    f"[{technique}]\n\n{explanation}\n\n"
                    "모든 기법 학습을 마쳤습니다. 이제 상담 모드로 전환해도 되겠습니까?😊 원하신다면 수면에 대한 고민을 입력해주세요."
                )
                state["mode"] = "상담 모드"
            state["history"].append((None, msg))
            return state["history"], state
        else:
            state["mode"] = "상담 모드"
            state["history"].append(
                (None, "모든 기법 학습을 마쳤습니다. 이제 상담 모드로 전환해도 되겠습니까?😊 원하신다면 수면에 대한 고민을 입력해주세요.")
            )
            return state["history"], state
    else:
        state["history"].append((None, "아무 내용이라도 입력해 주세요."))
        return state["history"], state

def consult_mode(user_input, state):
    chat = state["history"]

    # Iterative advice
    if state.get("iterative_advice_active", False):
        feedback = user_input.strip()
        chat.append((user_input, None))
        if feedback.lower() == "만족":
            final_advice = state.get("current_iterative_advice", "")
            state["iterative_advice_active"] = False
            state["awaiting_termination"] = True
            chat.append((None, "대화를 종료하고 싶으신가요? '예' 또는 '아니오'로 답해주세요."))
            return chat, state
        else:
            state["iterative_context"] += " " + feedback
            last_advice = state.get("current_iterative_advice", "")
            tech_name = ENG_TO_KOR_KEY.get(state["recommended_approach"], "수면 기법")
            new_advice = generate_personalized_advice(feedback, last_advice, tech_name)
            state["current_iterative_advice"] = new_advice
            chat.append(
                (None, new_advice + "\n추가 의견이 있으시면 입력해 주세요. 만족하시면 '만족'을 입력해 주세요.")
            )
            return chat, state

    # Termination handling
    if state.get("awaiting_termination", False):
        ans = normalize_yes_no(user_input) or user_input.strip()
        chat.append((user_input, None))
        if ans == "예":
            msg = save_history_to_json(state)
            chat.append((None, f"감사합니다. 대화를 종료합니다. {msg}"))
            state = reset_state(state)
            return chat, state
        elif ans == "아니오":
            chat.append((None, "대화를 계속 진행합니다. 어떤 고민이 있으신가요?"))
            state["awaiting_termination"] = False
            return chat, state
        else:
            chat.append((None, "대화를 종료하고 싶으신가요? '예' 또는 '아니오'로 답해주세요."))
            return chat, state

    if user_input.strip().lower() == "exit":
        msg = save_history_to_json(state)
        chat.append((None, f"대화가 종료되었습니다. {msg} 다시 시작하고 싶으시면 닉네임을 입력해주세요."))
        state = reset_state(state)
        return chat, state

    if state.get("waiting_end_confirmation", False):
        ans = normalize_yes_no(user_input) or user_input.strip()
        if ans == "예":
            msg = save_history_to_json(state)
            chat.append((None, f"감사합니다. 대화를 종료합니다. {msg}"))
            state = reset_state(state)
            return chat, state
        elif ans == "아니오":
            chat.append((None, "대화를 계속 진행합니다. 어떤 고민이 있으신가요?"))
            state["waiting_end_confirmation"] = False
            return chat, state
        else:
            chat.append((None, "대화를 종료하고 싶으신가요? '예' 또는 '아니오'로 답해주세요."))
            return chat, state

    # Self-decision
    if state.get("self_decision_pending", False):
        ans = normalize_yes_no(user_input) or user_input.strip()
        if ans == "예":
            state["iterative_advice_active"] = True
            state["iterative_context"] = state["consult_query"] + " " + merge_hints_to_utterance(state["socratic_hints"])
            info = retrieve_info_by_approach(state["iterative_context"], state["recommended_approach"], df, state)
            initial_advice = generate_response(
                state["iterative_context"], state["recommended_approach"], info
            )
            state["current_iterative_advice"] = initial_advice
            chat.append(
                (None, initial_advice + "\n추가 의견이 있으시면 입력해 주세요. 만족하시면 '만족'을 입력해 주세요.")
            )
            state["self_decision_pending"] = False
            return chat, state
        elif ans == "아니오":
            response_text = (
                "알겠습니다. 아래 기법들 중 하나를 사용하기 원하신다면 번호(1~5)를 입력해 주세요:\n"
                "1. 수면 제한 요법 (Sleep Restriction)\n"
                "2. 자극 조절 요법 (Stimulus Control)\n"
                "3. 수면 위생 교육 (Sleep Hygiene)\n"
                "4. 이완 요법 (Relaxation Techniques)\n"
                "5. 인지적 재구성 (Cognitive Restructuring)"
            )
            chat.append((None, response_text))
            state["self_decision_pending"] = False
            state["technique_selection_pending"] = True
            return chat, state
        else:
            chat.append((None, "‘예’ 또는 ‘아니오’로 답해 주세요."))
            return chat, state

    # Technique selection
    if state.get("technique_selection_pending", False):
        ans = user_input.strip()
        if ans in ["1", "2", "3", "4", "5"]:
            matched_key = list(ENG_TO_KOR_KEY.keys())[int(ans) - 1]
            state["recommended_approach"] = matched_key
            state["iterative_advice_active"] = True
            state["iterative_context"] = state["consult_query"] + " " + merge_hints_to_utterance(state["socratic_hints"])
            info = retrieve_info_by_approach(
                state["iterative_context"], state["recommended_approach"], df, state
            )
            initial_advice = generate_response(
                state["iterative_context"], state["recommended_approach"], info, include_termination=False
            )
            state["current_iterative_advice"] = initial_advice
            chat.append(
                (None, initial_advice + "\n추가 의견이 있으시면 입력해 주세요. 만족하시면 '만족'을 입력해 주세요.")
            )
            state["technique_selection_pending"] = False
            return chat, state
        else:
            chat.append(
                (
                    None,
                    "잘못된 번호입니다. 아래 번호 중 하나를 입력해 주세요:\n"
                    "1. 수면 제한 요법 (Sleep Restriction)\n"
                    "2. 자극 조절 요법 (Stimulus Control)\n"
                    "3. 수면 위생 교육 (Sleep Hygiene)\n"
                    "4. 이완 요법 (Relaxation Techniques)\n"
                    "5. 인지적 재구성 (Cognitive Restructuring)",
                )
            )
            return chat, state

    # Initial consult start
    if not state["consulting_active"]:
        eng_input = translate_to_english(user_input)
        state.update(
            {
                "consulting_active": True,
                "socratic_active": True,
                "consult_query": eng_input,
                "socratic_depth": 0,
                "socratic_hints": [],
            }
        )
        subq = generate_full_subquestion_v2(eng_input, depth=0, state=state)
        state["current_subquestion"] = subq
        chat.append((user_input, None))
        chat.append((None, subq))
        return chat, state

    # Ongoing consult
    chat.append((user_input, None))
    eng_input = translate_to_english(user_input)
    state["socratic_hints"].append((state["current_subquestion"], eng_input))
    state["socratic_depth"] += 1

    if state["current_confidence"] == "high" or state["socratic_depth"] >= state["max_depth"]:
        decision_msg = generate_self_decision_message(state)
        chat.append((None, decision_msg))
        state["self_decision_pending"] = True
        return chat, state

    ctx = state["consult_query"] + " " + merge_hints_to_utterance(state["socratic_hints"])
    subq = generate_full_subquestion_v2(eng_input, ctx, depth=state["socratic_depth"], state=state)
    state["current_subquestion"] = subq
    chat.append((None, subq))
    return chat, state

# =========================
# 9. Gradio UI (No Public Share by default)
# =========================
def user_input_handler(user_input, state):
    if state.get("mode") == "학습 모드":
        history, state = process_learning_mode(user_input, state)
        return history, "", state
    else:
        history, state = consult_mode(user_input, state)
        return history, "", state

def chatbot_entry(nickname, state):
    state = reset_state(state)
    state["nickname"] = nickname.strip() if nickname and nickname.strip() else None
    state["mode"] = "학습 모드"
    learning_msg = (
        "수면을 개선하는 데 도움이 되는 다섯 가지 기법이 있어요. 하나씩 함께 설명을 드려도 될까요?😊 "
        "계속 진행하려면 '예'를 입력해 주세요."
    )
    state["history"].append((None, f"반갑습니다! {learning_msg}"))
    return state["history"], state, gr.update(visible=False)

with gr.Blocks(css=".gradio-container { width: 80% !important; }") as demo:
    chatbot = gr.Chatbot(label="수면 인지 행동 치료 챗봇", bubble_full_width=True)
    name_input = gr.Textbox(label="닉네임(선택)", placeholder="입력하지 않아도 됩니다.")
    start_button = gr.Button("대화 시작")
    user_input = gr.Textbox(label="메시지를 입력해주세요.")
    session_state = gr.State(get_initial_state())

    start_button.click(
        fn=chatbot_entry, inputs=[name_input, session_state], outputs=[chatbot, session_state, name_input]
    )

    user_input.submit(
        fn=user_input_handler, inputs=[user_input, session_state], outputs=[chatbot, user_input, session_state]
    )

demo.launch(share=GRADIO_SHARE)
