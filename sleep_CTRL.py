# ctrl_chatbot.py (ANONYMIZED)
import gradio as gr
from openai import OpenAI
import os
import time
import json
import random
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datetime import datetime
import re
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import uuid

# ==================== 연구자 설정 ====================
# 실험 조건 기본값 (환경변수로 override 가능): EXP / CTRL
ASSIGNED_ARM = os.getenv("ASSIGNED_ARM", "CTRL").upper().strip()

# ==================== 0. OpenAI & HF 설정 ====================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment variables.")
client = OpenAI(api_key=OPENAI_API_KEY)

HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
if HF_ACCESS_TOKEN is None:
    raise ValueError("Missing HF_ACCESS_TOKEN in environment variables.")

# 모델 이름(가명) - 환경변수로 교체 가능
ROBERTA_MODEL_NAME = os.getenv("ROBERTA_MODEL_NAME", "orgname/roberta-cbti-finetuned-anon")

# ==================== 1. 분류 모델 로드 ====================
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL_NAME, use_auth_token=HF_ACCESS_TOKEN)
    model = RobertaForSequenceClassification.from_pretrained(ROBERTA_MODEL_NAME, use_auth_token=HF_ACCESS_TOKEN).to(device)
    model.eval()
    LABELS = ["Sleep Hygiene", "Stimulus Control", "Sleep Restriction", "Relaxation Techniques", "Cognitive Restructuring"]
    print("✅ RoBERTa intent classifier loaded.")
except Exception as e:
    raise RuntimeError(f"Model load error: {e}")

# ==================== 2. RAG 데이터/임베딩 ====================
RAG_DATA_FILE = os.getenv("RAG_DATA_FILE", "").strip()

def _load_rag_df(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    # 원본 스키마를 안전하게 매핑
    # 예상 컬럼: ['User Utterance (English)', 'Approach', 'Therapist Response (English)', 'Rejection Message', 'Intent']
    # 불일치 시 컬럼명을 추정하여 매핑
    cols = {c.lower(): c for c in df.columns}
    def get_col(*cands):
        for c in cands:
            if c.lower() in cols:
                return cols[c.lower()]
        return None
    user_c = get_col("User Utterance (English)", "user_input")
    approach_c = get_col("Approach", "approach")
    info_c = get_col("Therapist Response (English)", "info")
    rej_c = get_col("Rejection Message", "rejection_message")
    intent_c = get_col("Intent", "intent")
    out = pd.DataFrame({
        "user_input": df[user_c] if user_c else "",
        "approach": df[approach_c] if approach_c else "",
        "info": df[info_c] if info_c else "",
        "rejection_message": df[rej_c] if rej_c else "",
        "intent": df[intent_c] if intent_c else "",
    })
    out = out.dropna(subset=["user_input", "approach", "info"])
    return out

def _sample_rag_df() -> pd.DataFrame:
    # 안전한 예시 데이터(민감정보 없음)
    rows = [
        {"user_input": "I lie awake for hours.", "approach": "Sleep Restriction", "info": "Limit time in bed to consolidate sleep efficiency.", "rejection_message": "", "intent": "info"},
        {"user_input": "I use my phone in bed.", "approach": "Stimulus Control", "info": "Reserve bed for sleep; get up if unable to sleep.", "rejection_message": "", "intent": "info"},
        {"user_input": "I drink coffee late.", "approach": "Sleep Hygiene", "info": "Reduce late caffeine; keep regular sleep/wake times.", "rejection_message": "", "intent": "info"},
        {"user_input": "I feel tense at bedtime.", "approach": "Relaxation Techniques", "info": "Practice slow breathing and PMR before bed.", "rejection_message": "", "intent": "info"},
        {"user_input": "If I don't sleep, tomorrow is ruined.", "approach": "Cognitive Restructuring", "info": "Challenge catastrophic thoughts; find balanced alternatives.", "rejection_message": "", "intent": "info"},
    ]
    return pd.DataFrame(rows)

try:
    if RAG_DATA_FILE and os.path.exists(RAG_DATA_FILE):
        df_rag = _load_rag_df(RAG_DATA_FILE)
    else:
        df_rag = _sample_rag_df()
    embedding_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
    print("✅ RAG table & embedding model ready.")
except Exception as e:
    raise RuntimeError(f"RAG load error: {e}")

# ==================== 3. 단계/프롬프트/데이터 ====================
STAGE_NAME_INPUT = "name_input"
STAGE_PSYCHOEDUCATION_START = "psychoeducation_start"
STAGE_PSYCHOEDUCATION = "psychoeducation"
STAGE_PROBLEM_CONFIRM = "problem_confirm"
STAGE_USER_CONFIRMATION = "user_confirmation"
STAGE_WAIT_FOR_SOCRATIC_START = "wait_for_socratic_start"
STAGE_SOCRATIC_QUESTIONING = "socratic_questioning"
STAGE_MICRO_PE_AND_RQ1 = "micro_pe_and_rq1"
STAGE_RQ2_PLANNING = "rq2_planning"
STAGE_FINAL_PLAN_CONFIRM = "final_plan_confirm"

CBT_I_DESCRIPTIONS = {
    "Sleep Hygiene": "수면 위생 교육은 건강한 수면을 위해 생활 습관을 개선하는 방법입니다. 예: 늦은 카페인 제한, 일정한 취침·기상 시간 유지, 취침 전 전자기기 사용 줄이기 등.",
    "Stimulus Control": "자극 조절 요법은 침대를 오직 수면만을 위한 장소로 인식하게 돕습니다. 잠이 오지 않으면 즉시 침대에서 벗어나고 졸릴 때만 돌아갑니다.",
    "Sleep Restriction": "수면 제한 요법은 침대에 머무는 시간을 실제 수면 시간과 맞춰 시작해 점차 늘려가며 수면 효율을 높이는 기법입니다.",
    "Relaxation Techniques": "이완 요법은 심리적·신체적 긴장을 낮춰 자연스러운 수면을 돕습니다. 심호흡, 명상, 가벼운 스트레칭 등.",
    "Cognitive Restructuring": "인지적 재구성은 수면 관련 부정적 자동사고를 균형 잡힌 생각으로 재구성하여 불안을 낮춥니다.",
}
KOR_LABELS = {
    "수면 위생": "Sleep Hygiene",
    "자극 조절": "Stimulus Control",
    "수면 제한": "Sleep Restriction",
    "이완": "Relaxation Techniques",
    "인지 재구성": "Cognitive Restructuring",
}

PROMPTS_EXP = {
    "rq1_explore_intro": """
당신은 공감적 수면 상담사입니다. 사용자 발언: '{{user_input}}'과 관련된 맥락: '{{context}}'을 바탕으로, 사용자가 자신의 수면 문제에 대해 스스로 더 깊이 탐색하고 인식할 수 있도록 돕는 탐색형 질문을 1~2문장으로 만들어주세요. 공감하는 문장과 함께, 수면의 중요성에 대한 한두 문장 정도의 심리교육을 포함하세요.
""",
    "socratic_question_generator": """
당신은 따뜻하고 공감 능력이 뛰어난 수면 인지 행동 치료(CBT-I) 전문가입니다.
사용자가 자신의 수면 문제에 대해 더 깊이 고민하고 스스로 인식할 수 있도록 돕는 소크라테스식 질문을 만들어야 합니다.
사용자의 이전 발언: '{{user_input}}'
대화 맥락: '{{context}}'
지정된 질문 유형: '{{question_type}}'
사용자의 발언에 대한 공감과 함께, 지정된 질문 유형에 맞는 질문을 한 문장으로 만들어주세요.
""",
    "micro_pe_after_socratic": """
사용자의 발언과 대화 맥락을 기반으로, 사용자의 문제가 '{{label}}' 기법과 관련이 있음을 설명하세요.
직접적인 조언보다는, '{{label}}'이 사용자가 겪는 어려움에 어떻게 도움이 될 수 있는지 공감하는 톤으로 연결해주세요.
사용자가 자신의 상황을 새로운 관점에서 인식하고, 다음 단계의 행동 계획에 대해 스스로 생각해 볼 수 있도록 유도하세요.
""",
    "rq2_self_decision_rejection_handler": """
사용자가 제시된 행동 계획 옵션에 대해 '{{rejection_reason}}'과 같은 이유로 어려움이나 불확실성을 표현했습니다.
사용자의 감정을 공감하는 문장을 만들고,
'어떤 종류의 계획이 더 도움이 될 것 같나요? 아니면 다른 방식으로 계획을 세워볼까요?'와 같이 사용자에게 더 적합한 계획을 함께 탐색하도록 유도하는 질문을 1~2문장으로 제시하세요。
대화 맥락: '{{context}}'
""",
    "rq2_self_decision_followup_question": """
사용자는 행동 계획 옵션에 대해 '{{user_input}}'과 같이 추가적인 질문이나 반응을 보였습니다.
사용자의 질문에 대해 친절하고 구체적으로 답변해주세요. 답변 후, '혹시 다른 궁금한 점이 있으신가요?'와 같이 대화를 이어갈 수 있는 질문으로 마무리하세요.
대화 맥락: '{{context}}'
""",
    "rq2_escalated_rejection_message": """
제안드린 계획들이 현재 당신에게는 부담스럽거나 적절하지 않다고 느끼시는군요. 그런 마음이 드는 것은 충분히 이해가 됩니다. 지금 당장 어떤 행동을 정하는 것이 어렵다면, 잠시 쉬어가거나 다른 방식으로 고민을 이어가는 것도 괜찮아요. 대화를 마무리할까요? (예/아니오)
""",
    "rq2_rag_enhanced_prompt": """
당신은 수면 행동 계획 전문가입니다. 사용자의 문제는 주로 **{{predicted_label}}** 기법과 관련이 있습니다.
사용자의 핵심 문제점은 다음과 같습니다: {{problem_summary}}
다음은 RAG 시스템에서 검색된 관련 정보입니다:
{{retrieved_info}}
위 정보를 바탕으로 사용자에게 가장 도움이 될 만한 구체적인 행동 계획 옵션 3가지를 '①', '②', '③'과 같이 명확한 번호로 제시하세요. 계획은 사용자의 핵심 문제와 연관되어야 합니다.
옵션 제시 후, '그리고 이 중에서 어떤 것을 선택하시겠어요? 왜 그걸 고르셨는지 한 줄로 적어주세요.'라고 질문하여 사용자가 직접 선택하고 그 이유를 말하게 유도하세요.
""",
    "rq2_alternative_offer": """
네, 물론입니다. 다른 방법도 함께 고민해 볼 수 있어요. {{predicted_label}} 외에 또 다른 수면 기법에 대해 더 알아볼까요? 아니면 어떤 점이 마음에 들지 않으셨는지 더 자세히 이야기해 주실 수 있나요?
"""
}

PROMPTS_CTRL = {
    "rq1_inform_intro": """
당신은 수면 건강에 대한 정보를 간결하게 제공하는 상담사입니다. 사용자 발언: '{{user_input}}'과 관련된 맥락: '{{context}}'을 바탕으로, 불면증의 정의, 일반적인 지표 또는 수면에 미치는 영향에 대해 2~3문장으로 간략하게 안내하세요.
""",
    "analysis_and_pe_and_rq1": """
당신은 수면 상담 전문가입니다. 사용자 문제의 원인으로 '{{label}}'이 확정되었습니다.
이 '{{label}}' 카테고리에 대해 2~3문장으로 간결한 심리교육(Micro-PE)을 제공하세요.
""",
    "rq1_inform_question": """
혹시 최근에 스트레스나 생활 패턴의 변화가 있었나요? 아니면 일상에서 어떤 수면 환경을 조성하고 계신가요?
""",
    "rq2_directive_command": """
당신은 수면 행동 계획 전문가입니다. 사용자의 문제는 **{{predicted_label}}** 기법과 관련이 있습니다. 이 문제에 가장 적합한 **구체적인 행동 계획을 오직 한 가지**만 명확하게 지시하는 문장으로 만들어주세요.

예시 답변:
- 오늘 밤부터 잠들기 전 30분 동안 스마트폰 사용을 멈추고 책을 읽어보세요.
- 매일 아침 같은 시간에 일어나 햇빛을 쬐어보세요.

제안 후, '이 행동을 오늘 실천해 보시겠어요?'라고 질문하여 사용자의 확인을 요청하세요.
""",
    "rq2_rejection_and_alternative_offer": """
사용자가 제안된 행동 계획에 대해 거부 의사를 표현했습니다.
사용자의 거부 이유나 감정에 공감하는 짧은 문장을 만들고, 현재 제안한 {{predicted_label}} 기법 외에 다른 수면 기법(예: 자극 조절, 수면 제한)도 있다는 점을 간단히 언급하며, 세션을 부드럽게 마무리하는 메시지를 1-2문장으로 만들어주세요.
""",
    "rq2_directive_final_message_accept": """
좋습니다. 당신의 수면 개선을 응원합니다. 대화를 마무리하겠습니다.
""",
    "rq2_directive_final_message_reject": """
네, 알겠습니다. 다른 방법을 시도하고 싶으신 것 같네요. 대화를 마무리하겠습니다.
""",
    "rq2_directive_final_message_no_intent": """
그렇군요. 오늘은 계획을 정하기 어려우신 것 같네요. 대화를 마무리하겠습니다.
"""
}

PROMPTS_COMMON = {
    "socratic_type_selector": """
당신은 소크라테스식 질문 유형 분류 전문가입니다.
다음 5가지 유형 중 사용자 발언에 가장 적합한 유형을 선택하세요:
- clarity
- assumptions
- reasons_evidence
- implication_consequences
- alternate_viewpoints_perspectives

사용자 발언: '{{user_input}}'
대화 맥락: '{{context}}'

다른 설명 없이 오직 유형의 영어 이름만 출력하세요.
""",
    "confidence_check": """
당신은 사용자의 발언을 분석하는 AI입니다. 다음 대화 맥락과 사용자 발언을 고려했을 때, 사용자의 수면 문제 원인이 명확하게 파악되었다고 얼마나 확신하나요?
사용자 발언: '{{user_input}}'
대화 맥락: '{{context}}'
'high' / 'middle' / 'low' 중 하나만 출력하세요.
""",
    "rq2_problem_summary": """
당신은 수면 상담 전문가입니다. 사용자의 문제점을 공감하며 간결하게 한 문장으로 요약하세요.
대화 맥락: '{{context}}'
""",
    "final_plan_confirm": """
당신은 사용자의 계획을 지지하고 격려하는 상담사입니다. 사용자가 선택한 계획과 그 이유를 바탕으로, 매우 짧고 긍정적인 격려 메시지를 한 문장으로 작성하세요.
""",
    "translate_ko_to_en": """
주어진 한국어 문장을 영어로 번역하세요. 번역 외에는 다른 말을 하지 마세요.
""",
    "rq2_self_decision_intent_classifier": """
당신은 사용자의 행동 계획 선택에 대한 의도를 분류하는 AI입니다.
- 'rejection_doubt'
- 'acceptance_selection'
사용자 발언: '{{user_input}}'
대화 맥락: '{{context}}'
오직 라벨만 영어로 출력하세요.
""",
    "rq2_directive_rejection_handler": """
사용자가 제시된 행동 계획을 거부했습니다. 사용자의 거부 이유: '{{rejection_reason}}'을 공감하는 문장을 만들고,
'그렇다면 어떤 점이 부담스러웠나요? 다른 방법은 없을지 함께 고민해 볼까요?'라는 질문을 1~2문장으로 제시하세요.
대화 맥락: '{{context}}'
""",
    "rq2_user_intent_classifier": """
당신은 사용자의 발언 의도를 분류하는 AI입니다.
- 'request_alternatives'
- 'direct_rejection'
- 'agreement_or_elaboration'
사용자 발언: '{{user_input}}'
대화 맥락: '{{context}}'
오직 라벨만 영어로 출력하세요.
"""
}

# ==================== 4. 유틸 ====================
YES = {"예", "네", "응", "맞아요", "좋아요", "그래요", "넵"}
NO = {"아니오", "아니요", "아뇨", "싫어요", "원치않아요", "노"}

def yn_intent(text):
    t = text.strip().replace(" ", "")
    if "싫어" in t or any(n in t for n in NO):
        return "N"
    if any(y in t for y in YES):
        return "Y"
    return None

def count_reason_sentences(text: str) -> int:
    sentences = re.split(r'[.!?]\s*', text)
    return sum(1 for s in sentences if len(s.strip()) > 8 and any(k in s for k in ["왜", "때문", "이유"]))

def count_goal_sentences(text: str) -> int:
    sentences = re.split(r'[.!?]\s*', text)
    return sum(1 for s in sentences if any(k in s for k in ["하겠", "해볼게", "않겠", "실천", "노력"]))

def count_plan_sentences(text: str) -> int:
    sentences = re.split(r'[.!?]\s*', text)
    return sum(1 for s in sentences if len(s.strip()) > 8)

def assign_conditions(state):
    state["arm"] = ASSIGNED_ARM if ASSIGNED_ARM in {"EXP", "CTRL"} else "CTRL"
    if state["arm"] == "EXP":
        state["rq1_mode"] = "explore"
        state["rq2_mode"] = "self_decision"
        state["rq3_mode"] = "exploratory"
        state["max_socratic_depth"] = 4
    else:
        state["rq1_mode"] = "inform"
        state["rq2_mode"] = "directive"
        state["rq3_mode"] = "prescriptive"
        state["max_socratic_depth"] = 1
    log_interaction(state["log_file_path"], f"[실험배정] ARM={state['arm']}, RQ1={state['rq1_mode']}, RQ2={state['rq2_mode']}, RQ3={state['rq3_mode']}", tag="assign")

def setup_logging(state):
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = state.get("session_id") or uuid.uuid4().hex
    state["session_id"] = session_id
    log_filename = f"{log_dir}/session_{session_id}_{ts}.txt"
    state['log_file_path'] = log_filename
    state['start_time'] = time.time()
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(f"--- session start: {ts} ---\n\n")
    print(f"✅ Log file: {log_filename}")

def log_interaction(log_file_path, message, tag=None):
    if log_file_path:
        timestamp = datetime.now().strftime("%H:%M:%S")
        tag_str = f"[{tag}] " if tag else ""
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {tag_str}{message}\n")

def call_gpt_api(system_prompt_content, user_message_content, model="gpt-4o-mini", conversation_history=None):
    messages = [{"role": "system", "content": system_prompt_content}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message_content})
    try:
        completion = client.chat.completions.create(model=model, messages=messages)
        return completion.choices[0].message.content
    except Exception as e:
        log_interaction(None, f"GPT API error: {e}")
        return "죄송해요, 잠시 문제가 발생했습니다. 다시 시도해 주세요."

def select_socratic_type(user_input, context, state):
    prompt = PROMPTS_COMMON["socratic_type_selector"].replace("{{user_input}}", user_input).replace("{{context}}", context)
    response = call_gpt_api(prompt, user_input)
    valid = ["clarity", "assumptions", "reasons_evidence", "implication_consequences", "alternate_viewpoints_perspectives"]
    selected = (response or "").strip().lower()
    if selected not in valid:
        selected = "clarity"
    log_interaction(state.get('log_file_path'), f"[시스템] 소크라테스 유형: {selected}", tag="socratic_type")
    return selected

def generate_socratic_question(user_input, context, question_type, state):
    prompt = PROMPTS_EXP["socratic_question_generator"].replace("{{user_input}}", user_input).replace("{{context}}", context).replace("{{question_type}}", question_type)
    response = call_gpt_api(prompt, user_input)
    log_interaction(state.get('log_file_path'), f"[시스템] 소크라테스 질문: {response}", tag="socratic_gen")
    return response

def classify_with_bert(text, state):
    # 한국어 → 영어 번역(원문 로그 미저장)
    translated_text = call_gpt_api(PROMPTS_COMMON["translate_ko_to_en"], text)
    inputs = tokenizer(translated_text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        conf, predicted_idx = torch.max(probs, dim=1)
    label = LABELS[predicted_idx.item()]
    conf_val = conf.item()
    HIGH_CONF_TH = 0.55
    confidence_status = "높음" if conf_val >= HIGH_CONF_TH else "낮음"
    log_interaction(state.get('log_file_path'), f"[시스템] BERT 라벨={label}, 신뢰도={confidence_status}", tag="bert_classify")
    return label, confidence_status, probs.squeeze()

def retrieve_rag_info(query, approach, state, top_k=1):
    if 'faiss_index_cache' not in state:
        state['faiss_index_cache'] = {}
    if approach not in state['faiss_index_cache']:
        filtered_df = df_rag[df_rag['approach'] == approach].copy()
        if filtered_df.empty:
            return ""
        embeddings = embedding_model.encode(filtered_df['info'].tolist(), convert_to_tensor=True).cpu().numpy()
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        state['faiss_index_cache'][approach] = (index, filtered_df)
    index, filtered_df = state['faiss_index_cache'][approach]
    # 질의도 영어로 통일
    translated_query = call_gpt_api(PROMPTS_COMMON["translate_ko_to_en"], query)
    query_vec = embedding_model.encode([translated_query], convert_to_tensor=True).cpu().numpy()
    _, indices = index.search(query_vec, top_k)
    if indices.size > 0:
        return filtered_df.iloc[indices[0][0]]['info']
    return ""

def export_session_summary(state, final_message):
    summary_dir = "./session_summaries"
    os.makedirs(summary_dir, exist_ok=True)
    end_time = time.time()
    elapsed_seconds = end_time - state.get('start_time', end_time)
    summary = {
        "session_id": state.get('session_id'),
        "arm": state.get('arm'),
        "predicted_label": state.get('predicted_label'),
        "rq1_mode": state.get('rq1_mode'),
        "rq2_mode": state.get('rq2_mode'),
        "rq3_mode": state.get('rq3_mode'),
        "socratic_turns": state.get('socratic_turns', 0),
        "inform_turns": state.get('inform_turns', 0),
        "directive_accepts": state.get('directive_accepts', 0),
        "directive_denies": state.get('directive_denies', 0),
        "rq2_rejection_count": state.get('rq2_rejection_count', 0),
        "total_turns": len(state.get('history', [])) / 2,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "session_ended_at": datetime.now().isoformat(),
        "final_message": final_message,
    }
    summary_filename = f"{summary_dir}/session_{state['session_id']}.jsonl"
    with open(summary_filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(summary, ensure_ascii=False) + '\n')
    print(f"✅ Session summary: {summary_filename}")

def _end_session(state, chat_history, user_input, final_message, psychoeducation_row):
    final_plan_response = call_gpt_api(PROMPTS_COMMON["final_plan_confirm"], final_message, conversation_history=chat_history)
    chat_history.append({"role": "assistant", "content": final_plan_response})
    log_interaction(state.get('log_file_path'), f"[챗봇]: {final_plan_response}", tag="final_plan_confirm")
    final_label = state.get('predicted_label', '미분류')
    log_interaction(state.get('log_file_path'), f"[시스템] 최종 CBT-I: {final_label}", tag="final_label")
    state['session_ended_at'] = datetime.now().isoformat()
    log_interaction(state.get('log_file_path'), f"[시스템] 세션 종료", tag="session_end")
    export_session_summary(state, final_plan_response)
    state['stage'] = STAGE_FINAL_PLAN_CONFIRM
    return "", chat_history, state, gr.update(visible=False)

def classify_user_plan_intent(user_input, context, state):
    prompt = PROMPTS_COMMON["rq2_self_decision_intent_classifier"].replace("{{user_input}}", user_input).replace("{{context}}", context)
    response = call_gpt_api(prompt, user_input)
    res = (response or "").strip().lower()
    log_interaction(state.get('log_file_path'), f"[시스템] 계획 의도 분류: {res}", tag="plan_intent")
    return res

def classify_user_micro_pe_intent(user_input, context, state):
    prompt = PROMPTS_COMMON["rq2_user_intent_classifier"].replace("{{user_input}}", user_input).replace("{{context}}", context)
    response = call_gpt_api(prompt, user_input)
    res = (response or "").strip().lower()
    log_interaction(state.get('log_file_path'), f"[시스템] Micro-PE 의도: {res}", tag="user_intent_pe")
    return res

# ==================== 5. 메인 핸들러 ====================
def user_input_handler(user_input, state):
    new_state = state.copy()
    chat_history = new_state.get('history', [])
    current_stage = new_state.get('stage', STAGE_NAME_INPUT)

    if current_stage not in [STAGE_NAME_INPUT, STAGE_PSYCHOEDUCATION_START]:
        chat_history.append({"role": "user", "content": user_input})
        log_interaction(new_state.get('log_file_path'), f"[사용자]: <redacted>", tag="user_input")  # 원문 미기록

    if current_stage == STAGE_PSYCHOEDUCATION_START:
        new_state['stage'] = STAGE_PSYCHOEDUCATION
        bot_message = "아래 버튼으로 기법 설명을 볼 수 있어요. 또는 '준비'를 입력해 바로 시작할 수 있어요."
        chat_history.append({"role": "assistant", "content": bot_message})
        log_interaction(new_state.get('log_file_path'), f"[챗봇]: {bot_message}", tag="pe_start")
        return "", chat_history, new_state, gr.update(visible=True)

    elif current_stage == STAGE_PSYCHOEDUCATION:
        if user_input == "준비":
            new_state['stage'] = STAGE_PROBLEM_CONFIRM
            bot_message = "좋습니다. 어제나 오늘, 잠과 관련해 가장 불편했던 점이 있었을까요?"
            chat_history.append({"role": "assistant", "content": bot_message})
            log_interaction(new_state.get('log_file_path'), f"[챗봇]: {bot_message}", tag="problem_confirm")
            return "", chat_history, new_state, gr.update(visible=False)
        elif user_input in KOR_LABELS.keys():
            english_label = KOR_LABELS[user_input]
            bot_message = CBT_I_DESCRIPTIONS[english_label]
            chat_history.append({"role": "assistant", "content": bot_message})
            log_interaction(new_state.get('log_file_path'), f"[챗봇]: <pe_info>", tag="pe_info")
            bot_message_2 = "다른 기법도 보실 수 있어요. 준비되셨다면 '준비'라고 입력해 주세요."
            chat_history.append({"role": "assistant", "content": bot_message_2})
            return "", chat_history, new_state, gr.update(visible=True)
        else:
            bot_message = "버튼을 누르거나 '준비'를 입력해 주세요."
            chat_history.append({"role": "assistant", "content": bot_message})
            return "", chat_history, new_state, gr.update(visible=True)

    elif current_stage == STAGE_PROBLEM_CONFIRM:
        new_state['initial_problem_statement'] = user_input
        if new_state["arm"] == "EXP":
            explore_prompt = PROMPTS_EXP["rq1_explore_intro"].replace("{{user_input}}", user_input).replace("{{context}}", "")
            bot_message = call_gpt_api(explore_prompt, user_input, conversation_history=chat_history)
            chat_history.append({"role":"assistant","content":bot_message})
            new_state['stage'] = STAGE_WAIT_FOR_SOCRATIC_START
            new_state['socratic_turns'] = 1
            return "", chat_history, new_state, gr.update(visible=False)
        else:
            label, _, _ = classify_with_bert(user_input, new_state)
            new_state['predicted_label'] = label
            pe_and_q = (
                "당신은 수면 상담 전문가입니다. 사용자 발언: '{{user_input}}'을 바탕으로,\n"
                "1. 사용자의 문제 원인으로 '{{label}}'이 확정되었음을 알리고 이에 대한 2-3문장의 간결한 심리교육(Micro-PE)을 제공하세요.\n"
                "2. 이후, 자연스럽게 사용자에게 '혹시 최근에 스트레스나 생활 패턴의 변화가 있었나요? "
                "아니면 일상에서 어떤 수면 환경을 조성하고 계신가요?'와 같은 후속 질문을 이어붙이세요.\n"
            )
            combined_prompt = pe_and_q.replace("{{user_input}}", user_input).replace("{{label}}", label)
            bot_message = call_gpt_api(combined_prompt, user_input, conversation_history=chat_history)
            chat_history.append({"role": "assistant", "content": bot_message})
            new_state['inform_turns'] = 1
            new_state['stage'] = STAGE_RQ2_PLANNING
            return "", chat_history, new_state, gr.update(visible=False)

    elif current_stage == STAGE_WAIT_FOR_SOCRATIC_START:
        new_state['socratic_session_initial_input'] = user_input
        new_state['socratic_hints'] = []
        context_for_init = (new_state.get('initial_problem_statement', '') + " " + user_input).strip()
        selected_type = select_socratic_type(user_input, context_for_init, new_state)
        bot_message = generate_socratic_question(user_input, context_for_init, selected_type, new_state)
        chat_history.append({"role": "assistant", "content": bot_message})
        new_state['stage'] = STAGE_SOCRATIC_QUESTIONING
        new_state['socratic_turns'] = new_state.get('socratic_turns', 0) + 1
        return "", chat_history, new_state, gr.update(visible=False)

    elif current_stage == STAGE_SOCRATIC_QUESTIONING:
        new_state['socratic_hints'].append(user_input)
        parts = [new_state.get('initial_problem_statement', '')]
        if new_state.get('socratic_session_initial_input'):
            parts.append(new_state['socratic_session_initial_input'])
        parts.extend(new_state.get('socratic_hints', []))
        full_context = " ".join(parts).strip()
        conf_prompt = PROMPTS_COMMON['confidence_check'].replace("{{user_input}}", user_input).replace("{{context}}", full_context)
        confidence = (call_gpt_api(conf_prompt, user_input) or "low").strip().lower()
        log_interaction(new_state.get('log_file_path'), f"[시스템] 신뢰도: {confidence}", tag="confidence")
        max_depth = new_state.get("max_socratic_depth")
        if confidence == 'high' or len(new_state['socratic_hints']) >= max_depth:
            combined_query = full_context
            label, _, _ = classify_with_bert(combined_query, new_state)
            new_state['predicted_label'] = label
            pe_prompt = PROMPTS_EXP["micro_pe_after_socratic"].replace("{{label}}", label)
            bot_message = call_gpt_api(pe_prompt, user_input, conversation_history=chat_history)
            chat_history.append({"role": "assistant", "content": bot_message})
            new_state['stage'] = STAGE_MICRO_PE_AND_RQ1
        else:
            selected_type = select_socratic_type(user_input, full_context, new_state)
            bot_message = generate_socratic_question(user_input, full_context, selected_type, new_state)
            chat_history.append({"role": "assistant", "content": bot_message})
            new_state['stage'] = STAGE_SOCRATIC_QUESTIONING
            new_state['socratic_turns'] = new_state.get('socratic_turns', 0) + 1
        return "", chat_history, new_state, gr.update(visible=False)

    elif current_stage == STAGE_MICRO_PE_AND_RQ1:
        parts = [new_state.get('initial_problem_statement', '')]
        if new_state.get('socratic_session_initial_input'):
            parts.append(new_state['socratic_session_initial_input'])
        parts.extend(new_state.get('socratic_hints', []))
        problem_summary_text = " ".join(parts).strip()
        summary_prompt = PROMPTS_COMMON["rq2_problem_summary"].replace("{{context}}", problem_summary_text)
        problem_summary = call_gpt_api(summary_prompt, "")
        if new_state["arm"] == "EXP":
            user_intent = classify_user_micro_pe_intent(user_input, problem_summary_text, new_state)
            retrieved_rag_info = retrieve_rag_info(problem_summary_text, new_state['predicted_label'], new_state)
            prompt_content = PROMPTS_EXP["rq2_rag_enhanced_prompt"].replace("{{predicted_label}}", new_state['predicted_label']).replace("{{problem_summary}}", problem_summary).replace("{{retrieved_info}}", retrieved_rag_info)
            bot_message = call_gpt_api(prompt_content, user_input, conversation_history=chat_history)
            chat_history.append({"role": "assistant", "content": bot_message})
            new_state['stage'] = STAGE_RQ2_PLANNING
            return "", chat_history, new_state, gr.update(visible=False)
        else:
            pe_question_prompt = PROMPTS_CTRL["rq1_inform_question"]
            bot_message = call_gpt_api(pe_question_prompt, user_input, conversation_history=chat_history)
            chat_history.append({"role": "assistant", "content": bot_message})
            new_state['stage'] = STAGE_RQ2_PLANNING
            return "", chat_history, new_state, gr.update(visible=False)

    elif current_stage == STAGE_RQ2_PLANNING:
        new_state['reasons_count'] += count_reason_sentences(user_input)
        new_state['goals_count'] += count_goal_sentences(user_input)
        new_state['plan_sentences_count'] += count_plan_sentences(user_input)
        log_interaction(new_state.get('log_file_path'),
                        f"[시스템] counters: 이유={new_state['reasons_count']}, 목표={new_state['goals_count']}, 계획문장={new_state['plan_sentences_count']}",
                        tag="counter")
        parts = [new_state.get('initial_problem_statement', '')]
        if new_state.get('socratic_session_initial_input'):
            parts.append(new_state['socratic_session_initial_input'])
        if new_state.get('socratic_hints'):
            parts.extend(new_state['socratic_hints'])
        problem_summary = " ".join(parts).strip()

        if new_state["arm"] == "EXP":
            user_int = classify_user_plan_intent(user_input, problem_summary, new_state)
            if user_int == 'rejection_doubt':
                new_state['rq2_rejection_count'] = new_state.get('rq2_rejection_count', 0) + 1
                if new_state['rq2_rejection_count'] >= 2:
                    msg_content = PROMPTS_EXP["rq2_escalated_rejection_message"]
                    bot_message = call_gpt_api(msg_content, user_input, conversation_history=chat_history)
                    chat_history.append({"role": "assistant", "content": bot_message})
                    new_state['stage'] = STAGE_FINAL_PLAN_CONFIRM
                    return "", chat_history, new_state, gr.update(visible=False)
                else:
                    rej_prompt = PROMPTS_EXP["rq2_self_decision_rejection_handler"].replace("{{rejection_reason}}", user_input).replace("{{context}}", problem_summary)
                    bot_message = call_gpt_api(rej_prompt, user_input, conversation_history=chat_history)
                    chat_history.append({"role": "assistant", "content": bot_message})
                    new_state['stage'] = STAGE_RQ2_PLANNING
                    return "", chat_history, new_state, gr.update(visible=False)
            else:
                if "?" in user_input or "어떤 종류" in user_input or "어떻게" in user_input:
                    followup = PROMPTS_EXP["rq2_self_decision_followup_question"].replace("{{user_input}}", user_input).replace("{{context}}", problem_summary)
                    bot_message = call_gpt_api(followup, user_input, conversation_history=chat_history)
                    chat_history.append({"role": "assistant", "content": bot_message})
                    new_state['stage'] = STAGE_RQ2_PLANNING
                    return "", chat_history, new_state, gr.update(visible=False)
                else:
                    return _end_session(new_state, chat_history, user_input, user_input, gr.update(visible=False))

        else:  # CTRL
            prompt_content = PROMPTS_CTRL["rq2_directive_command"].replace("{{predicted_label}}", new_state['predicted_label'])
            bot_message = call_gpt_api(prompt_content, user_input, conversation_history=chat_history)
            chat_history.append({"role": "assistant", "content": bot_message})
            new_state['stage'] = STAGE_FINAL_PLAN_CONFIRM
            return "", chat_history, new_state, gr.update(visible=False)

    elif current_stage == STAGE_FINAL_PLAN_CONFIRM:
        intent = yn_intent(user_input)
        if intent == "Y":
            final_message = PROMPTS_CTRL["rq2_directive_final_message_accept"]
            new_state['directive_accepts'] += 1
            return _end_session(new_state, chat_history, user_input, final_message, gr.update(visible=False))
        elif intent == "N":
            new_state['directive_denies'] += 1
            rejection_prompt = PROMPTS_CTRL["rq2_rejection_and_alternative_offer"].replace("{{predicted_label}}", new_state['predicted_label'])
            final_message = call_gpt_api(rejection_prompt, user_input, conversation_history=chat_history)
            return _end_session(new_state, chat_history, user_input, final_message, gr.update(visible=False))
        else:
            final_message = PROMPTS_CTRL["rq2_directive_final_message_no_intent"]
            return _end_session(new_state, chat_history, user_input, final_message, gr.update(visible=False))

    return "", chat_history, new_state, gr.update(visible=False)

# ==================== 6. Gradio UI ====================
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"

with gr.Blocks(css=".gradio-container { max_width: 800px; margin: auto; }") as demo:
    gr.Markdown("# 수면 인지 행동 치료 챗봇 💤")
    state = gr.State({
      'stage': STAGE_NAME_INPUT, 'history': [], 'log_file_path': None,
      'arm': None, 'rq1_mode': None, 'rq2_mode': None, 'rq3_mode': None, 'assignment_seed': None,
      'reasons_count': 0, 'goals_count': 0, 'plan_sentences_count': 0,
      'socratic_turns': 0, 'inform_turns': 0, 'directive_accepts': 0, 'directive_denies': 0,
      'participant_id': None, 'survey_token': None,
      'policy_version': 'v1.0',
      'session_started_at': None,
      'session_ended_at': None,
      'initial_problem_statement': None,
      'socratic_session_initial_input': None,
      'socratic_hints': [],
      'predicted_label': None,
      'rq2_rejection_count': 0,
      'faiss_index_cache': {},
      'session_id': uuid.uuid4().hex,
    })

    with gr.Row(visible=True) as intro_row:
        name_input = gr.Textbox(label="닉네임(선택)", placeholder="입력하지 않아도 됩니다.")
        name_submit_btn = gr.Button("시작하기")

    chatbot = gr.Chatbot(height=450, type='messages', visible=False)
    msg = gr.Textbox(placeholder="여기에 메시지를 입력하세요.", visible=False)

    with gr.Row(visible=False) as psychoeducation_row:
        edu_buttons_container = gr.Row()
        with edu_buttons_container:
            for label_key in KOR_LABELS.keys():
                btn = gr.Button(label_key, elem_id=f"edu_btn_{label_key}")
                btn.click(
                    fn=user_input_handler,
                    inputs=[gr.State(label_key), state],
                    outputs=[msg, chatbot, state, psychoeducation_row]
                )
            edu_ready_btn = gr.Button("준비", elem_id="edu_btn_ready")
            edu_ready_btn.click(
                fn=user_input_handler,
                inputs=[gr.State("준비"), state],
                outputs=[msg, chatbot, state, psychoeducation_row]
            )

    def start_chat_wrapper(name, state_obj):
        new_state = state_obj.copy()
        # 닉네임은 상태에만 저장(로그/파일명에 포함하지 않음)
        new_state['user_name'] = name.strip() if name and name.strip() else None
        new_state['participant_id'] = f"pid_{int(time.time())}"  # 파일명/요약파일에 사용하지 않음
        new_state['session_started_at'] = datetime.now().isoformat()
        new_state['survey_token'] = new_state['participant_id']

        setup_logging(new_state)
        assign_conditions(new_state)

        new_state['history'].append({"role": "assistant", "content": "반갑습니다! 간단히 살펴보고 시작해볼게요."})
        new_state['stage'] = STAGE_PSYCHOEDUCATION_START
        return "", new_state['history'], new_state, gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)

    name_submit_btn.click(
        fn=start_chat_wrapper,
        inputs=[name_input, state],
        outputs=[name_input, chatbot, state, intro_row, msg, chatbot, psychoeducation_row]
    ).then(
        fn=user_input_handler,
        inputs=[gr.State(""), state],
        outputs=[msg, chatbot, state, psychoeducation_row]
    )

    msg.submit(
        fn=user_input_handler,
        inputs=[msg, state],
        outputs=[msg, chatbot, state, psychoeducation_row]
    )

if __name__ == "__main__":
    demo.launch(share=GRADIO_SHARE)
