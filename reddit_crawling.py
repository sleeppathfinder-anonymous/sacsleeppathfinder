# -*- coding: utf-8 -*-
# Converted from Jupyter Notebook on 2025-10-18T03:57:57
# Markdown cells are preserved as comments.


# ==== Cell 1 (code) ====
import praw
import pandas as pd

# Reddit API 자격 증명
reddit = praw.Reddit(
    client_id='',       # 실제 클라이언트 ID로 대체
    client_secret='', # 실제 클라이언트 Secret으로 대체
    user_agent=''      # 사용자 에이전트 이름
)

# Subreddit 데이터를 크롤링하는 함수
def crawl_subreddit(subreddit_name, limit=1000):
    try:
        subreddit = reddit.subreddit(subreddit_name)
        posts = []
        for post in subreddit.hot(limit=limit):  # 인기 게시물 검색
            posts.append({
                'Subreddit': subreddit_name,
                'Title': post.title,
                'Score': post.score,
                'URL': post.url,
                'Comments': post.num_comments,
                'Selftext': post.selftext,
                'Author': post.author.name if post.author else 'Deleted'
            })
        print(f"{subreddit_name}: {len(posts)} posts crawled.")
        return pd.DataFrame(posts)
    except Exception as e:
        print(f"Error crawling {subreddit_name}: {e}")
        return pd.DataFrame()  # 빈 DataFrame 반환

# 크롤링할 subreddit 목록
subreddits = ['insomnia', 'sleepdisorders', 'sleep']

# 각 subreddit에서 데이터를 크롤링하고 병합
all_data = pd.DataFrame()
for subreddit_name in subreddits:
    subreddit_data = crawl_subreddit(subreddit_name)
    all_data = pd.concat([all_data, subreddit_data], ignore_index=True)

# 데이터 정제
all_data = all_data[all_data['Selftext'].str.contains("sleep", case=False, na=False)]  # "sleep" 단어가 포함된 글만 필터링
all_data = all_data[all_data['Author'] != 'Deleted']  # 삭제된 작성자 제외
all_data.drop_duplicates(subset=['Title', 'Selftext'], inplace=True)  # 중복된 글 제거

# 데이터 저장
output_file = '../data/<redacted_token>.csv'
all_data.to_csv(output_file, index=False)
print(f"Data saved to {output_file}")

# 크롤링 데이터 통계 출력
print("=== Crawling Summary ===")
print(f"Total posts: {len(all_data)}")
print(all_data['Subreddit'].value_counts())
print(all_data.head())

# ==== Cell 2 (code) ====
import pandas as pd

# 기존 병합된 데이터 불러오기 (파일 경로에 맞게 수정)
df_combined = pd.read_csv("../data/<redacted_token>.csv")

# 새롭게 수집한 데이터 불러오기
df_new = pd.read_csv("../data/<redacted_token>.csv")

# 기존 데이터에 없는 URL을 가진 행만 추출
df_unique = df_new[~df_new['URL'].isin(df_combined['URL'])]

# 결과 확인
df_unique

# ==== Cell 3 (code) ====
df_unique.to_csv("../data/<redacted_token>.csv", index=False)

# ==== Cell 4 (code) ====
# 병합된 데이터 불러오기
df_combined = pd.read_csv("../data/<redacted_token>.csv")

# 'selftext' 컬럼에서 'sleep' 단어 포함 여부 확인 후 필터링
df_filtered = df_combined[df_combined['Selftext'].str.contains(r'\bsleep\b', case=False, na=False)]

# 결과 확인
df_filtered

# ==== Cell 5 (code) ====
# 필요하면 저장
df_filtered.to_csv("../data/<redacted_token>.csv", index=False)

# ==== Cell 6 (code) ====
import pandas as pd

# 병합된 데이터 불러오기
df = pd.read_csv("../data/<redacted_token>.csv")
df

# ==== Cell 7 (code) ====
df2 = pd.read_csv(".../path/to/file")
df2

# ==== Cell 8 (code) ====
final_df = pd.concat([df, df2], axis=0)
final_df

# ==== Cell 9 (code) ====
# URL 기준 중복 제거 (처음 나타난 것만 남김)
df_unique = final_df.drop_duplicates(subset="URL", keep="first")

# 결과 확인
df_unique

# ==== Cell 10 (code) ====
df_unique.to_csv(".../path/to/file", index=False)

# ==== Cell 11 (code) ====
df_final = pd.read_csv(".../path/to/file")
df_final

# ==== Cell 12 (code) ====
import openai
import pandas as pd
from tqdm import tqdm
import re
import json

# ========== API 키 불러오기 ==========
def load_api_key(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

api_key_path = '../api_key_new.txt'
openai.api_key = load_api_key(api_key_path)

# ========== GPT 응답 생성 함수 ==========
def generate_response_gpt(prompt, model="gpt-4o-mini"):
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content":
                    "You are an empathetic and knowledgeable assistant who helps people with sleep problems express their concerns in a concise and natural way."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,
            max_tokens=150,
            top_p=0.95
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ GPT API 호출 중 오류 발생: {e}")
        return "Error: API Issue"

# ========== 개인정보 제거 함수 ==========
def remove_personal_info(text):
    text = re.sub(r"\bI'?m \d{1,2}(-year-old)?\b", "I", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(as a )?(male|female|man|woman)\b", "", text, flags=re.IGNORECASE)
    medication_names = ["Wellbutrin", "Ambien", "Melatonin", "Trazodone"]
    for med in medication_names:
        text = re.sub(r"\b" + re.escape(med) + r"\b", "[medication]", text, flags=re.IGNORECASE)
    return text.strip()

# ========== 개선된 Intent 분류 함수 ==========
def classify_cbt_intent(user_utterance, retry_limit=3):
    intent_prompt = f"""
You are a CBT-I (Cognitive Behavioral Therapy for Insomnia) expert helping categorize sleep-related concerns into one of five therapeutic categories.

The following is a user’s sleep-related concern:
"{user_utterance}"

Classify it into **exactly one** of the following five CBT-I categories:

1. Sleep Restriction: The concern describes spending excessive time in bed, fragmented sleep, oversleeping, or waking up too early.
   Example: "I go to bed at 9 PM but can’t fall asleep until 2 AM. Then I wake up again at 6 AM."

2. Stimulus Control: The concern describes lying in bed awake for long periods or being mentally alert in bed. The bed is associated with frustration, thinking, or other non-sleep activities.
   Example: "I spend hours in bed trying to fall asleep, just tossing and turning."

3. Cognitive Restructuring: The concern involves negative thoughts, exaggerated fears, or anxieties about sleep (e.g., catastrophizing, hopelessness).
   Example: "If I don’t sleep tonight, I’ll mess up my entire life."

4. Sleep Hygiene: The concern is about poor habits, irregular routines, food, drinks, screen time, or environmental factors.
   Example: "I drink 3 cups of coffee at night and stay on my phone until 2 AM."

5. Relaxation Techniques: The concern involves strong emotional tension, mental overactivity, or inability to calm down due to stress.
   Example: "Even when I’m tired, my mind won’t stop racing because of work stress."

Rules:
- Respond ONLY in the following JSON format: {{ "Intent": "<One of the five categories above>" }}
- DO NOT invent new categories.
- DO NOT return multiple categories.
"""

    valid_intents = {
        "Sleep Restriction",
        "Stimulus Control",
        "Cognitive Restructuring",
        "Sleep Hygiene",
        "Relaxation Techniques"
    }

    for attempt in range(retry_limit):
        response = generate_response_gpt(intent_prompt)

        try:
            parsed = json.loads(response)
            intent = parsed.get("Intent", "").strip()
            if intent in valid_intents:
                return intent
            else:
                print(f"⚠️ [시도 {attempt+1}] 유효하지 않은 Intent 응답: {intent}")
        except json.JSONDecodeError:
            print(f"⚠️ [시도 {attempt+1}] JSON 파싱 실패:\n{response}")
        except Exception as e:
            print(f"⚠️ [시도 {attempt+1}] 예외 발생: {e}")

    print("⚠️ 최대 재시도 도달 - 기본값 Cognitive Restructuring 반환")
    return "Cognitive Restructuring"

# ========== Reddit 데이터 처리 함수 ==========
def process_reddit_data(df):
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        selftext = row["Selftext"]
        if pd.isna(selftext) or len(selftext.strip()) == 0:
            continue

        prompt = f"""
Below is a sleep-related concern expressed by a user:
"{selftext}"

Rewrite this as if the user is directly expressing their concern in a one-on-one conversation with a sleep therapist.
Ensure that:
- The original concern and emotion of the user are maintained.
- The statement is personal, realistic, and natural.
- Any personally identifiable information (age, gender, occupation, location, specific personal experiences) is **removed or generalized**.
- The language is suitable for a professional consultation setting, avoiding casual or community-style phrasing (e.g., 'Has anyone experienced this?').
- The length remains similar but unnecessary repetitions are removed.
"""

        user_utterance = generate_response_gpt(prompt)
        user_utterance = remove_personal_info(user_utterance)
        intent = classify_cbt_intent(user_utterance)

        results.append({
            "User Utterance": user_utterance,
            "Intent": intent,
            "Original Selftext": selftext
        })

    return pd.DataFrame(results)

# ========== 실행부 ==========
df_reddit = pd.read_csv(".../path/to/file")
df_processed = process_reddit_data(df_reddit)
df_processed.to_csv(".../path/to/file", index=False)
print("✅ 데이터 생성 완료! → final_intent_data_0413.csv' 저장됨.")

# ==== Cell 13 (code) ====
import pandas as pd
import glob
import os

# 전체 파일 불러오기
file_pattern = ".../path/to/file*.csv"
file_list = glob.glob(file_pattern)

# 파일명에서 숫자 추출 후 11 이상인 파일만 필터링
filtered_files = [
    f for f in file_list
    if int(os.path.basename(f).split("_")[-1].split(".")[0]) >= 11
]

# 숫자 순 정렬
filtered_files = sorted(filtered_files, key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]))

# 병합
df_merged = pd.concat([pd.read_csv(f) for f in filtered_files], ignore_index=True)

# 저장
df_merged.to_csv("../data/<redacted_token>.csv", index=False)
print(f"✅ 병합 완료! → <redacted_token>.csv (총 {len(df_merged)} rows)")

# ==== Cell 14 (code) ====
df_merged['Intent'].value_counts()

# ==== Cell 15 (code) ====
import openai
import pandas as pd
import json
from tqdm import tqdm

# 🔑 OpenAI API 키 불러오기
openai.api_key = open("../api_key_new.txt").read().strip()

# ✅ 증강 대상 intent
target_intents = {
    "Sleep Restriction": 2000,
    "Sleep Hygiene": 2000,
    "Relaxation Techniques": 2000,
    "Stimulus Control": 2000  
}

# ✅ intent별 영어 설명 (for GPT prompt)
intent_english_desc = {
    "Sleep Restriction": "Spending too much time in bed compared to actual sleep time, or fragmented sleep patterns.",
    "Stimulus Control": "Lying in bed while awake, or associating the bed with frustration or alertness.",
    "Cognitive Restructuring": "Expressing negative thoughts, excessive worries, or hopelessness about sleep.",
    "Sleep Hygiene": "Poor sleep habits or environmental factors such as caffeine, screen time, or irregular routines.",
    "Relaxation Techniques": "Stress, emotional tension, or mental overactivity interfering with the ability to relax before sleep."
}

# ✅ GPT paraphrasing prompt 생성 함수
def gpt_paraphrase_prompt(text, intent):
    return f"""
You are a CBT-I (Cognitive Behavioral Therapy for Insomnia) expert.

The following sentence expresses a user's sleep-related concern and belongs to the category '{intent}'.

Your task is to:
- Rewrite the sentence naturally, using different wording,
- While preserving the original meaning and emotional tone,
- And ensuring that it still clearly fits within the CBT-I category '{intent}'.

📘 Definition of '{intent}':
{intent_english_desc[intent]}

---

Original:
\"{text}\"

Paraphrased:
"""

# ✅ GPT 호출 함수
def generate_paraphrase(text, intent):
    prompt = gpt_paraphrase_prompt(text, intent)
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ GPT Error: {e}")
        return None

# ✅ 기존 데이터 로드
df = pd.read_csv("../data/<redacted_token>.csv")

# ✅ 증강 대상 추출 및 실행
augmented_rows = []

for intent, target_count in target_intents.items():
    current_df = df[df["Intent"] == intent]
    current_count = len(current_df)
    needed = target_count - current_count
    print(f"\n▶️ {intent}: {current_count}개 → {target_count}개로 {needed}개 생성 시작")

    if needed <= 0:
        continue

    # 증강 샘플 랜덤 추출 (복제 허용)
    sampled = current_df.sample(n=needed, replace=True, random_state=42)

    for _, row in tqdm(sampled.iterrows(), total=len(sampled)):
        orig_text = row["User Utterance"]
        new_text = generate_paraphrase(orig_text, intent)
        if new_text and new_text != orig_text:
            augmented_rows.append({
                "User Utterance": new_text,
                "Intent": intent,
                "Original Selftext": f"[AUG from] {orig_text}"
            })

# ✅ DataFrame 변환 및 저장
df_augmented = pd.DataFrame(augmented_rows)
df_final = pd.concat([df, df_augmented], ignore_index=True)

df_final.to_csv("../data/<redacted_token>.csv", index=False)
print("\n✅ GPT 기반 증강 완료 → <redacted_token>.csv")
print(f"총 샘플 수: {len(df_final)}")

# ==== Cell 16 (code) ====
df_final

# ==== Cell 17 (code) ====
df_final['Intent'].value_counts()

# ==== Cell 18 (code) ====
import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv("../data/<redacted_token>.csv")
# 데이터 확인
df

# ==== Cell 19 (code) ====
import openai
import pandas as pd
from tqdm import tqdm
import re

# GPT API 키 불러오기
def load_api_key(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

api_key_path = '../api_key_new.txt'  # API 키 경로
openai.api_key = load_api_key(api_key_path)

# GPT 모델 사용
def generate_response_gpt(prompt, model="gpt-4o-mini"):
    """
    Generates a response based on the given prompt using GPT.
    """
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content":
                    "You are an empathetic and knowledgeable assistant who helps people with sleep problems express their concerns in a concise and natural way. "
                    "Your task is to rewrite user concerns in a way that feels personal and realistic, as if a real person with insomnia is expressing their thoughts. "
                    "\n\n"
                    "- Maintain the original concern and emotion of the user.\n"
                    "- Avoid slang, internet jargon, or community-style phrases (e.g., 'your guys,' 'lol,' 'wait for the birds!').\n"
                    "- Ensure the utterance is conversational and natural, making it suitable for a conversation with a sleep therapist.\n"
                    "- Remove community-based phrasing like 'Has anyone experienced this?' and instead phrase it as a direct concern or question to a professional.\n"
                    "- Keep the length similar but remove unnecessary repetitions."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,
            max_tokens=150,
            top_p=0.95
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ GPT API 호출 중 오류 발생: {e}")  # 오류 메시지를 콘솔에 출력
        return "Error: API Issue"  # 오류 발생 시 기본값 반환

def remove_personal_info(text):
    """
    사용자의 발화에서 개인 정보를 제거하는 함수
    (나이, 성별, 이름, 특정 약물명 등)
    """
    # 나이 관련 표현 제거 (예: "I'm 37", "I'm a 25-year-old male")
    text = re.sub(r"\bI'?m \d{1,2}(-year-old)?\b", "I", text, flags=re.IGNORECASE)

    # 성별 관련 표현 제거 (예: "I'm a male/female", "as a woman, I feel...")
    text = re.sub(r"\b(as a )?(male|female|man|woman)\b", "", text, flags=re.IGNORECASE)

    # 특정 약물명 제거 (Wellbutrin, Ambien 등)
    medication_names = ["Wellbutrin", "Ambien", "Melatonin", "Trazodone"]
    for med in medication_names:
        text = re.sub(r"\b" + re.escape(med) + r"\b", "[medication]", text, flags=re.IGNORECASE)

    return text.strip()

# Intent (CBT-I 요소) 분류 함수
def classify_cbt_intent(user_utterance):
    """
    사용자 발화를 기반으로 CBT-I 요소 중 하나를 정확하게 분류
    (반드시 5가지 요소 중 하나로 분류되도록 제한)
    """
    intent_prompt = f"""
    The following is a sleep-related user utterance:
    "{user_utterance}"

    Classify the utterance into **one and only one** of the following CBT-I categories:

    - **Sleep Restriction** → If the user describes irregular sleep patterns, fragmented sleep, or sleeping too little/too much.
    - **Stimulus Control** → If the user describes lying in bed but struggling to fall asleep, or conditioned arousal related to their bed environment.
    - **Cognitive Restructuring** → If the user expresses negative thoughts or anxieties about sleep that may worsen insomnia.
    - **Sleep Hygiene** → If the issue is related to habits, food, beverages, or environmental factors affecting sleep.
    - **Relaxation Techniques** → If the issue is about frustration, anxiety, or emotional distress preventing sleep.

    **Rules:**
    - **Select only one category from the five listed above.**
    - **Do not create a new category.**
    - **Do not return multiple categories.**
    - **Always respond in the following format:**

    Intent: <Assigned Category>
    """

    response = generate_response_gpt(intent_prompt)

    valid_intents = {
        "Sleep Restriction", "Stimulus Control",
        "Cognitive Restructuring", "Sleep Hygiene", "Relaxation Techniques"
    }

    intent = "Unknown"
    for line in response.split("\n"):
        if "Intent:" in line:
            extracted_intent = line.split(":")[1].strip()
            if extracted_intent in valid_intents:  # 5가지 요소만 허용
                intent = extracted_intent

    # "Unknown"이면 Cognitive Restructuring으로 기본값 적용
    if intent == "Unknown":
        intent = "Cognitive Restructuring"

    return intent

# GPT 응답 생성 및 저장
def process_reddit_data(df):
    """
    Reddit 데이터에서 사용자 발화 생성 및 Intent 분류 후 저장 (개인정보 제거 및 전문가 상담 형식 변환)
    """
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        selftext = row["Selftext"]
        if pd.isna(selftext) or len(selftext.strip()) == 0:
            continue  # 빈 텍스트 제외

        # 사용자 발화 생성
        prompt = f"""
        Below is a sleep-related concern expressed by a user:
        "{selftext}"

        Rewrite this as if the user is directly expressing their concern in a one-on-one conversation with a sleep therapist.
        Ensure that:
        - The original concern and emotion of the user are maintained.
        - The statement is personal, realistic, and natural.
        - Any personally identifiable information (age, gender, occupation, location, specific personal experiences) is **removed or generalized**.
        - The language is suitable for a professional consultation setting, avoiding casual or community-style phrasing (e.g., 'Has anyone experienced this?').
        - The length remains similar but unnecessary repetitions are removed.
        """

        user_utterance = generate_response_gpt(prompt)

        # Intent (CBT-I 요소) 분류
        intent = classify_cbt_intent(user_utterance)

        # 결과 저장
        results.append({
            "User Utterance": user_utterance,
            "Intent": intent,
            "Original Selftext": selftext  # 원본 텍스트도 저장 (비교용)
        })

    return pd.DataFrame(results)

# Reddit 데이터 불러오기
df_reddit = pd.read_csv("../data/<redacted_token>.csv")

# 데이터 처리 및 생성
df_processed = process_reddit_data(df_reddit)

# 결과 저장
df_processed.to_csv("../data/<redacted_token>.csv", index=False)

print("🔹 데이터 생성 완료! '<redacted_token>.csv'로 저장되었습니다.")

# ==== Cell 20 (code) ====
import praw
import pandas as pd

# Reddit API 자격 증명
reddit = praw.Reddit(
    client_id='1JGOpZ3RZ5XytSR734KMmw',       # 실제 클라이언트 ID로 대체
    client_secret='<redacted_token>', # 실제 클라이언트 Secret으로 대체
    user_agent='InsomniaCrawler'      # 사용자 에이전트 이름
)

def <redacted_token>(subreddit_name, keyword, limit=1000):
    try:
        subreddit = reddit.subreddit(subreddit_name)
        posts = []
        for post in subreddit.search(query=keyword, limit=limit, sort='relevance'):
            posts.append({
                'Subreddit': subreddit_name,
                'Title': post.title,
                'Score': post.score,
                'URL': post.url,
                'Comments': post.num_comments,
                'Selftext': post.selftext,
                'Author': post.author.name if post.author else 'Deleted'
            })
        print(f"{subreddit_name} ({keyword}): {len(posts)} posts crawled.")
        return pd.DataFrame(posts)
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

# ==== Cell 21 (code) ====
# 특정 키워드로 필터링된 insomnia 글만 수집
df_stimulus = <redacted_token>("insomnia", "stimulus control", limit=1000)
df_stimulus = df_stimulus[df_stimulus['Author'] != 'Deleted']  # 삭제 제외
df_stimulus.drop_duplicates(subset=["Title", "Selftext"], inplace=True)

# 저장
df_stimulus.to_csv("../data/<redacted_token>.csv", index=False)
print("✅ 저장 완료: <redacted_token>.csv")

# ==== Cell 22 (code) ====
# 특정 키워드로 필터링된 insomnia 글만 수집
df_stimulus = <redacted_token>("sleep", "stimulus control", limit=1000)
df_stimulus = df_stimulus[df_stimulus['Author'] != 'Deleted']  # 삭제 제외
df_stimulus.drop_duplicates(subset=["Title", "Selftext"], inplace=True)

# 저장
df_stimulus.to_csv("../data/<redacted_token>.csv", index=False)
print("✅ 저장 완료: <redacted_token>.csv")

# ==== Cell 23 (code) ====
df_stimulus

# ==== Cell 24 (code) ====
import pandas as pd

df_combined = pd.concat([df, df2], axis=0)  # 세로로 합치기
df_combined

# ==== Cell 25 (code) ====
df_combined['Intent'].value_counts()

# ==== Cell 26 (code) ====
# 결과 저장
df_combined.to_csv("../data/<redacted_token>.csv", index=False)

# ==== Cell 27 (code) ====
import openai
import pandas as pd
import random

# 🔹 GPT API 키 불러오기
def load_api_key(file_path):
    """API 키를 파일에서 불러오는 함수"""
    with open(file_path, 'r') as file:
        return file.read().strip()

# API 키 파일 경로 (실제 경로로 변경할 것)
api_key_path = '../api_key.txt'  
openai.api_key = load_api_key(api_key_path)

# 🔹 샘플 데이터 (Stimulus Control & Relaxation Techniques)
sample_data = [
    {"text": "I keep waking up in the middle of the night.", "Intent": "Stimulus Control"},
    {"text": "I have trouble falling asleep on time.", "Intent": "Relaxation Techniques"},
    {"text": "I wake up too early and can't fall back asleep.", "Intent": "Stimulus Control"},
    {"text": "I feel anxious when trying to sleep.", "Intent": "Relaxation Techniques"},
    {"text": "I keep checking my phone before bed, and it keeps me awake.", "Intent": "Stimulus Control"}
]

# 🔹 GPT-4o를 활용한 데이터 증강 함수 (리스트로 여러 개 생성)
def generate_augmented_text(prompt, num_samples=3):
    """GPT-4o를 사용하여 문장을 변형하여 데이터 증강"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a chatbot that helps generate CBT-I user utterances."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=200  # 최대 토큰 늘리기 (여러 개 문장 포함 가능)
        )
        
        # 반환된 응답에서 내용 추출
        augmented_text = response.choices[0].message.content.strip()

        # 변형된 문장을 개행 문자("\n") 기준으로 분할
        augmented_sentences = augmented_text.split("\n")

        # 빈 문장 제거 및 최대 `num_samples` 개수만 반환
        return [s.strip() for s in augmented_sentences if s.strip()][:num_samples]

    except Exception as e:
        print(f"Error generating text: {e}")
        return []

# 🔹 샘플 데이터 증강 실행
augmented_samples = []
for sample in sample_data:
    prompt = f"Generate {3} different variations of the following user utteranc.\path\to\file
    new_texts = generate_augmented_text(prompt, num_samples=3)
    
    for new_text in new_texts:
        augmented_samples.append({"Original": sample["text"], "Augmented": new_text, "Intent": sample["Intent"]})

# 🔹 결과 출력
for i, sample in enumerate(augmented_samples):
    print(f"🔹 Original: {sample['Original']}")
    print(f"  ➡ Augmented: {sample['Augmented']}")
    print(f"  (Intent: {sample['Intent']})\n")

# ==== Cell 28 (code) ====
import openai
import pandas as pd
import random
import time
from tqdm import tqdm

# 🔹 GPT API 키 불러오기
def load_api_key(file_path):
    """API 키를 파일에서 불러오는 함수"""
    with open(file_path, 'r') as file:
        return file.read().strip()

# API 키 파일 경로 (실제 경로로 변경할 것)
api_key_path = '../api_key.txt'  
openai.api_key = load_api_key(api_key_path)

# 🔹 데이터 불러오기
df = pd.read_csv("../data/<redacted_token>.csv")

# 🔹 증강할 클래스 및 목표 개수 설정
target_counts = {
    "Sleep Restriction": 3000,
    "Cognitive Restructuring": 3000,
    "Sleep Hygiene": 3000,
    "Stimulus Control": 3000,
    "Relaxation Techniques": 3000
}

# 🔹 현재 클래스별 샘플 수 확인
current_counts = df["Intent"].value_counts().to_dict()

# 🔹 증강할 데이터 수 계산
augmentation_needed = {
    intent: target_counts[intent] - current_counts.get(intent, 0)
    for intent in target_counts
    if current_counts.get(intent, 0) < target_counts[intent]
}

print("\n📌 증강할 샘플 수:", augmentation_needed)

# 🔹 GPT-4o를 활용한 데이터 증강 함수
def generate_augmented_text(prompt, num_samples=3):
    """GPT-4o를 사용하여 문장을 변형하여 데이터 증강"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a chatbot that helps generate CBT-I user utterances."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=200  # 최대 토큰 늘리기
        )
        
        # 반환된 응답에서 내용 추출
        augmented_text = response.choices[0].message.content.strip()
        augmented_sentences = augmented_text.split("\n")

        # 빈 문장 제거 및 최대 `num_samples` 개수만 반환
        return [s.strip() for s in augmented_sentences if s.strip()][:num_samples]

    except Exception as e:
        print(f"⚠️ Error generating text: {e}")
        return []

# 🔹 최종 증강 데이터 저장 리스트
augmented_data = []

# 🔹 부족한 샘플 개수만큼 Intent별 증강 (Intent별 독립 처리)
for intent, num_needed in augmentation_needed.items():
    df_subset = df[df["Intent"] == intent]
    
    print(f"\n🚀 증강 시작: {intent} (필요한 샘플 수: {num_needed})")
    
    intent_data = []  # Intent별 데이터 저장
    with tqdm(total=num_needed, desc=f"Generating {intent}") as pbar:
        while len(intent_data) < num_needed:
            # 원본 문장에서 랜덤 선택
            sample = df_subset.sample(n=1).iloc[0]
            
            # GPT-4o에게 증강 요청
            prompt = f"Generate 3 different variations of the following user utteranc.\path\to\file
            new_texts = generate_augmented_text(prompt, num_samples=3)
            
            for new_text in new_texts:
                if len(intent_data) < num_needed:
                    intent_data.append({
                        "User Utterance": new_text,
                        "Intent": intent,
                        "Original Selftext": sample["Original Selftext"]
                    })
                    pbar.update(1)  # tqdm 업데이트
            
            time.sleep(1)  # API Rate Limit 방지
        
    print(f"✅ {intent} 증강 완료! 총 {len(intent_data)}개 생성됨.")
    
    # 생성된 데이터를 최종 리스트에 추가
    augmented_data.extend(intent_data)

# 🔹 DataFrame 변환 후 기존 데이터와 결합
df_augmented = pd.DataFrame(augmented_data)
df_combined = pd.concat([df, df_augmented])

# 🔹 증강된 데이터 저장
output_path = "../data/<redacted_token>.csv"
df_combined.to_csv(output_path, index=False)
print(f"\n✅ 데이터 증강 완료! 저장 경로: {output_path}")

# ==== Cell 29 (code) ====
import pandas as pd

# CSV 파일 불러오기
df_aug = pd.read_csv("../data/<redacted_token>.csv")
# 데이터 확인
df_aug

# ==== Cell 30 (code) ====
df_aug['Intent'].value_counts()