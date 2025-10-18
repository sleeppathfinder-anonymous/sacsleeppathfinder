# -*- coding: utf-8 -*-
# Generated from notebook on 2025-10-18T03:56:19
# Cells converted from Jupyter Notebook. Markdown cells are included as comments.

from __future__ import annotations


# ==== Cell 1 (code) ====
# === Jupyter-only: dCBT-I 로그 분석 파이프라인 (인자 불필요) ===
# 사용법:
# 1) BASE = "ctrl_log/와 exp_log/가 들어있는 상위 폴더" 로 변경
# 2) OUT = "결과 폴더" 변경(선택)
# 3) 이 셀 실행하면 CSV 3개가 OUT에 생성됨

from pathlib import Path
from datetime import datetime, timedelta
import re, pandas as pd, numpy as np

# <redacted_token>
# 경로 설정 (이 줄만 수정하세요)
# <redacted_token>
BASE = r"data"   # <<<<<<<<<<<<< 바꾸기
OUT  = r"data/analysis_out"  # <<<<<<<< 바꾸기(선택)
PLOTS = True  # 간단 시각화 원하면 True

# <redacted_token>
# 설정/정규식
# <redacted_token>
ENCODINGS = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"]

ACCEPT_RE = re.compile(
    r"(해볼게|해보겠|해보겠습니다|해보죠|해봐야겠|시도해보|시도해 볼|네[.!?]?$|넵|좋아요|그러죠|그럴게요|알겠|오케이|OK)",
    re.IGNORECASE
)
REFUSE_RE = re.compile(
    r"(안\s?할래|싫어|못\s?하겠|어려울|지금은\s?좀|나중에|보류|패스|그만|중단|다른\s?건\s?없나|딴\s?거)",
    re.IGNORECASE
)

QUESTION_TAGS = (
    "<redacted_token>",
    "bot_problem_confirm",
    "bot_question",
)
DIRECTIVE_TAGS = (
    "<redacted_token>",
    "rq2_prompt_directive",
    "final_plan_confirm",
)
SESSION_END_TAG = "session_end"

LINE_RE   = re.compile(r"^\[(?P<time>\d{2}:\d{2}:\d{2})\]\s+\[(?P<etype>[^\]]+)\]\s+(?P<rest>.*)$")
HEADER_RE = re.compile(r"^---\s*대화 시작:\s*(?P<stamp>\d{8}_\d{6})\s*\(사용자:\s*(?P<user>[^,]+),\s*PI.\path\to\file
ASSIGN_RE = re.compile(r"\[(?P<time>\d{2}:\d{2}:\d{2})\]\s+\[실험배정\]\s+ARM=(?P<arm>\w+),\s*RQ1=(?P<rq1>\w+),\s*RQ2=(?P<rq2>\w+),\s*RQ3=(?P<rq3>\w+)", re.IGNORECASE)

ROLE_USER_RE = re.compile(r"\[(사용자|USER|내담자)\]")
ROLE_BOT_RE  = re.compile(r"\[(챗봇|BOT|상담사|ASSISTANT)\]")
ROLE_SYS_RE  = re.compile(r"\[(시스템|SYSTEM)\]")
CONTENT_SPLIT_RE = re.compile(r":\s*", re.UNICODE)

# <redacted_token>
# 유틸/파서
# <redacted_token>
def read_text(path: Path) -> str:
    for enc in ENCODINGS:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return path.read_text(errors="ignore")

def parse_one_file(path: Path, condition: str) -> pd.DataFrame:
    text = read_text(path)
    lines = [ln.rstrip("\n\r") for ln in text.splitlines() if ln.strip()]

    nickname = pid = sess_start_ts = None
    arm = rq1 = rq2 = rq3 = None
    events = []

    for ln in lines:
        m = HEADER_RE.match(ln)
        if m:
            nickname = m.group("user").strip()
            pid = m.group("pid").strip()
            sess_start_ts = m.group("stamp")
            continue
        m = ASSIGN_RE.match(ln)
        if m:
            arm = m.group("arm").upper().strip()
            rq1 = m.group("rq1").strip()
            rq2 = m.group("rq2").strip()
            rq3 = m.group("rq3").strip()
            continue

    for ln in lines:
        m = LINE_RE.match(ln)
        if not m:
            continue
        time = m.group("time")
        etype = m.group("etype")
        rest  = m.group("rest").strip()

        role = "unknown"
        if ROLE_USER_RE.search(rest):
            role = "user"
        elif ROLE_BOT_RE.search(rest):
            role = "bot"
        elif ROLE_SYS_RE.search(rest):
            role = "system"

        content = rest
        sp = CONTENT_SPLIT_RE.split(rest, maxsplit=1)
        if len(sp) == 2:
            content = sp[1].strip()

        events.append({
            "session_file": path.name,
            "condition": condition,
            "time": time,
            "etype": etype,
            "role": role,
            "content": content,
            "nickname": nickname,
            "pid": pid,
            "sess_start_ts": sess_start_ts,
            "arm": arm, "rq1": rq1, "rq2": rq2, "rq3": rq3,
        })

    cols = ["session_file","condition","time","etype","role","content",
            "nickname","pid","sess_start_ts","arm","rq1","rq2","rq3"]
    return pd.DataFrame(events) if events else pd.DataFrame(columns=cols)

def load_all(base: Path) -> pd.DataFrame:
    dfs = []
    for condition, sub in [("CTRL","ctrl_log"), ("EXP","exp_log")]:
        folder = base / sub
        if not folder.exists():
            print(f"[WARN] not found: {folder}")
            continue
        for f in sorted(folder.glob("*.txt")):
            dfs.append(parse_one_file(f, condition))
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    df["__order"] = df.groupby("session_file").cumcount()
    return df

def <redacted_token>(sess_df: pd.DataFrame, sess_start_ts: str) -> float:
    if sess_df.empty or sess_df["time"].isna().all():
        return np.nan
    try:
        base_date = datetime.strptime(sess_start_ts, "%Y%m%d_%H%M%S").date()
    except Exception:
        base_date = datetime.today().date()

    prev_dt = None
    current_date = base_date
    abs_times = []
    for t in pd.to_datetime(sess_df["time"], format="%H:%M:%S", errors="coerce"):
        if pd.isna(t): 
            continue
        candidate = datetime.combine(current_date, t.time())
        if prev_dt is not None and candidate < prev_dt:
            current_date = current_date + timedelta(days=1)  # 자정 롤오버
            candidate = datetime.combine(current_date, t.time())
        abs_times.append(candidate)
        prev_dt = candidate

    if len(abs_times) >= 2:
        return float((abs_times[-1] - abs_times[0]).total_seconds())
    return np.nan

def word_count(s: str) -> int:
    return 0 if not isinstance(s, str) else len(re.findall(r"\S+", s))

def char_count(s: str) -> int:
    return 0 if not isinstance(s, str) else len(s)

# <redacted_token>
# 파이프라인 실행
# <redacted_token>
BASE = Path(BASE)
OUT = Path(OUT); OUT.mkdir(parents=True, exist_ok=True)

df = load_all(BASE)

# 세션 메타/턴수
sess_meta = (df.groupby("session_file")
               .agg(condition=("condition","first"),
                    nickname=("nickname","first"),
                    pid=("pid","first"),
                    arm=("arm","first"),
                    rq1=("rq1","first"),
                    rq2=("rq2","first"),
                    rq3=("rq3","first"),
                    sess_start_ts=("sess_start_ts","first"))
               .reset_index())

turns = (df.assign(is_user=df["role"].eq("user") & df["etype"].eq("user_input"),
                   is_bot =df["role"].eq("bot") & df["etype"].str.startswith("bot_"))
           .groupby("session_file")
           .agg(user_turns=("is_user","sum"),
                bot_turns=("is_bot","sum"),
                n_lines=("etype","size"))
           .reset_index())
turns["turns"] = turns["user_turns"] + turns["bot_turns"]
turns["empty_session"] = (turns["user_turns"]==0)

dur = (df.groupby("session_file")
         .apply(lambda s: <redacted_token>(s, s["sess_start_ts"].iloc[0]))
         .rename("duration_sec").reset_index())

sess_base = sess_meta.merge(turns, on="session_file", how="left").merge(dur, on="session_file", how="left")

# RQ1
rq1_rows = []
for sid, sess in df.groupby("session_file", sort=False):
    q_mask = sess["etype"].isin(QUESTION_TAGS) & sess["role"].eq("bot")
    n_bot_questions = int(q_mask.sum())

    first_q_idx = sess.index[q_mask].min() if q_mask.any() else None
    dir_mask = sess["etype"].isin(DIRECTIVE_TAGS)
    first_dir_idx = sess.index[dir_mask].min() if dir_mask.any() else None

    cause_wc = cause_cc = 0
    if first_q_idx is not None:
        start = first_q_idx + 1
        end = first_dir_idx if first_dir_idx is not None else sess.index.max()+1
        window = sess.loc[start:end-1]
        u_msgs = window[(window["role"]=="user") & (window["etype"]=="user_input")]["content"].dropna()
        cause_wc = int(sum(word_count(x) for x in u_msgs))
        cause_cc = int(sum(char_count(x) for x in u_msgs))

    rq1_rows.append({
        "session_file": sid,
        "rq1_bot_question_turns": n_bot_questions,
        "rq1_user_cause_words": cause_wc,
        "rq1_user_cause_chars": cause_cc,
    })
rq1_df = pd.DataFrame(rq1_rows)

# RQ2
def recompute_accept_refuse(sess: pd.DataFrame):
    dir_idx = sess.index[(sess["role"].eq("bot")) & (sess["etype"].isin(DIRECTIVE_TAGS))]
    if not len(dir_idx):
        return (False, False, np.nan, False, False)
    first_dir = dir_idx.min()
    after = sess.loc[first_dir+1:]
    user_after = after[(after["role"]=="user") & (after["etype"]=="user_input")]
    if user_after.empty:
        early_end = after["etype"].eq(SESSION_END_TAG).any() and (after.index.max() - first_dir) <= 2
        return (False, False, np.nan, early_end, True)
    first_user = user_after.iloc[0]
    txt = (first_user["content"] or "").strip()
    acc = bool(ACCEPT_RE.search(txt))
    ref = bool(REFUSE_RE.search(txt))
    tta = int(first_user.name - first_dir)
    early_end = after["etype"].eq(SESSION_END_TAG).any() and (after.index.max() - first_dir) <= 2
    return (acc, ref, tta, early_end, True)

rq2_rows = []
for sid, sess in df.groupby("session_file", sort=False):
    acc, ref, tta, ee, has_dir = recompute_accept_refuse(sess)
    rq2_rows.append({
        "session_file": sid,
        "has_directive": has_dir,
        "accepted": acc,
        "refused": ref,
        "turns_to_accept": tta,
        "early_end_within2": ee
    })
rq2_df = pd.DataFrame(rq2_rows)

# 세션 요약
session_summary = (sess_base
                   .merge(rq1_df, on="session_file", how="left")
                   .merge(rq2_df, on="session_file", how="left"))
session_summary["use_in_analysis"] = ~session_summary["empty_session"]
session_summary["sentiment_delta"] = np.nan

# 분석집단: 빈세션 제외 + 지시 포함
A = session_summary.query("use_in_analysis and has_directive").copy()

# 조건/참가자 집계
def q25(s): return pd.to_numeric(s, errors="coerce").quantile(0.25)
def q75(s): return pd.to_numeric(s, errors="coerce").quantile(0.75)
def mean_if_num(s): return pd.to_numeric(s, errors="coerce").mean()

cond_agg = (A.assign(tta_cond=lambda d: d["turns_to_accept"].where(d["accepted"]))
              .groupby("condition")
              .agg(
                  n_sessions=("session_file","nunique"),
                  mean_turns=("turns","mean"),
                  median_turns=("turns","median"),
                  iqr_turns_low=("turns", q25),
                  iqr_turns_high=("turns", q75),
                  mean_duration_s=("duration_sec","mean"),
                  median_duration_s=("duration_sec","median"),
                  rq1_qturns_mean=("rq1_bot_question_turns","mean"),
                  rq1_cause_words_mean=("rq1_user_cause_words","mean"),
                  accept_rate=("accepted","mean"),
                  mean_tta_if_accepted=("tta_cond", mean_if_num),
                  early_end_rate=("early_end_within2","mean"),
              ).reset_index())

part_agg = (A.groupby(["condition","nickname"])
              .agg(
                  n_sessions=("session_file","nunique"),
                  total_turns=("turns","sum"),
                  mean_turns=("turns","mean"),
                  accept_rate=("accepted","mean"),
                  mean_tta=("turns_to_accept","mean"),
                  mean_qturns=("rq1_bot_question_turns","mean"),
                  mean_cause_words=("rq1_user_cause_words","mean"),
                  early_end_rate=("early_end_within2","mean"),
                  total_duration_s=("duration_sec","sum"),
              ).reset_index())

# 저장
Path(OUT).mkdir(parents=True, exist_ok=True)
pd.DataFrame(session_summary).to_csv(Path(OUT, "sessions_summary.csv"), index=False, encoding="utf-8")
pd.DataFrame(cond_agg).to_csv(Path(OUT, "condition_aggregate.csv"), index=False, encoding="utf-8")
pd.DataFrame(part_agg).to_csv(Path(OUT, "participant_aggregate.csv"), index=False, encoding="utf-8")

print("[OK] CSV 저장:", Path(OUT, "sessions_summary.csv"), Path(OUT, "condition_aggregate.csv"), Path(OUT, "participant_aggregate.csv"))

# (선택) 간단 시각화
if PLOTS and not A.empty:
    import matplotlib.pyplot as plt
    A.boxplot(column="turns", by="condition"); plt.suptitle(""); plt.title("Session Turns by Condition")
    plt.xlabel("Condition"); plt.ylabel("Turns"); plt.show()

    acc = A.groupby("condition")["accepted"].mean().reset_index()
    plt.bar(acc["condition"], acc["accepted"]); plt.title("Acceptance Rate by Condition")
    plt.ylim(0,1); plt.ylabel("Rate"); plt.show()

    eer = A.groupby("condition")["early_end_within2"].mean().reset_index()
    plt.bar(eer["condition"], eer["early_end_within2"]); plt.title("Early End (≤2 turns after directive)")
    plt.ylim(0,1); plt.ylabel("Rate"); plt.show()

# 미리보기
display(cond_agg.round(3), part_agg.head(), session_summary.head())


# ==== Cell 2 (code) ====
import pandas as pd
import numpy as np

PATH = "data/analysis_out/participant_aggregate.csv"  # 경로 맞게 수정
P = pd.read_csv(PATH)

# 파생: 총 지속시간(분, 시간) 보기 편하게
P["total_duration_min"] = P["total_duration_s"] / 60.0
P["total_duration_hr"]  = P["total_duration_s"] / 3600.0

# 보기 좋게 정렬(옵션)
P = P.sort_values(["condition","nickname"]).reset_index(drop=True)

display(P.head(10))

# ==== Cell 3 (code) ====
P2 = P.copy()

# 수락한 세션이 없는 참가자의 mean_tta는 NaN -> 공백(표시용)
P2["mean_tta_display"] = P2["mean_tta"].apply(lambda x: "" if pd.isna(x) else x)

# (대안) 만약 수락 없을 때 0으로 표기하고 싶다면 위 줄 대신 아래 줄 사용:
# P2["mean_tta_display"] = P2["mean_tta"].fillna(0)

display(P2[["condition","nickname","n_sessions","accept_rate","mean_tta","mean_tta_display"]].head(15))

# ==== Cell 4 (code) ====
# 1) 총 지속시간 상위(잠재적 이상치)
top_dur = P2.sort_values("total_duration_s", ascending=False).head(10)
print("== 총 지속시간 상위 10 ==")
display(top_dur[["condition","nickname","n_sessions","total_duration_s","total_duration_min","total_duration_hr"]])

# 2) 조기 종료율 1.0인 참가자 (모든 세션이 directive 후 2턴 내 종료)
full_early_end = P2[np.isclose(P2["early_end_rate"], 1.0)]
print("== Early-end rate == 1.0 참가자 ==")
display(full_early_end[["condition","nickname","n_sessions","early_end_rate","total_duration_min"]])

# 3) 조건별 요약(지속시간/턴수 중앙값 같이 보기)
cond_summary = (P2.groupby("condition")
                .agg(
                    n_participants=("nickname","nunique"),
                    n_sessions=("n_sessions","sum"),
                    mean_turns=("mean_turns","mean"),
                    median_mean_turns=("mean_turns","median"),
                    mean_total_minutes=("total_duration_min","mean"),
                    median_total_minutes=("total_duration_min","median"),
                    mean_accept_rate=("accept_rate","mean"),
                    median_accept_rate=("accept_rate","median"),
                    mean_early_end_rate=("early_end_rate","mean"),
                    median_early_end_rate=("early_end_rate","median"),
                    mean_cause_words=("mean_cause_words","mean"),
                    median_cause_words=("mean_cause_words","median"),
                )
                .reset_index())
print("== 조건별 참가자 집계 요약 ==")
display(cond_summary.round(3))

# ==== Cell 5 (code) ====
import matplotlib.pyplot as plt

# 참여자 평균 턴수 박스플롯
fig = plt.figure()
P2.boxplot(column="mean_turns", by="condition")
plt.suptitle("")
plt.title("Participant-level Mean Turns by Condition")
plt.ylabel("Mean Turns"); plt.xlabel("Condition")
plt.show()

# Acceptance rate 막대(평균)
acc = P2.groupby("condition")["accept_rate"].mean().reset_index()
plt.figure()
plt.bar(acc["condition"], acc["accept_rate"])
plt.title("Participant-level Acceptance Rate (mean)")
plt.ylim(0,1); plt.ylabel("Rate"); plt.show()

# 원인설명 단어수 박스플롯
fig = plt.figure()
P2.boxplot(column="mean_cause_words", by="condition")
plt.suptitle("")
plt.title("Participant-level Cause Words by Condition")
plt.ylabel("Words"); plt.xlabel("Condition")
plt.show()

# ==== Cell 6 (code) ====
import pandas as pd
import numpy as np

SESS_PATH = "data/analysis_out/sessions_summary.csv"  # 경로만 맞춰줘

S = pd.read_csv(SESS_PATH)

# ---- 타입/결측 보정 ----
bool_cols = ["empty_session","has_directive","accepted","refused","early_end_within2","use_in_analysis"]
for c in bool_cols:
    if c in S.columns:
        S[c] = S[c].astype("bool")

# turns_to_accept: '' -> NaN -> float
if "turns_to_accept" in S.columns:
    S["turns_to_accept"] = pd.to_numeric(S["turns_to_accept"], errors="coerce")

# sentiment_delta: 비어있을 수 있음
if "sentiment_delta" in S.columns:
    S["sentiment_delta"] = pd.to_numeric(S["sentiment_delta"], errors="coerce")

# sess_start_ts 문자열 → datetime (형식: YYYYMMDD_HHMMSS)
def parse_ts(s):
    try:
        return pd.to_datetime(s, format="%Y%m%d_%H%M%S")
    except Exception:
        return pd.NaT

if "sess_start_ts" in S.columns:
    S["sess_dt"] = S["sess_start_ts"].apply(parse_ts)
else:
    S["sess_dt"] = pd.NaT

# ---- 파생 지표 ----
S["duration_min"] = S["duration_sec"] / 60.0
S["q_turn_ratio"] = S["rq1_bot_question_turns"] / S["turns"].replace(0, np.nan)
S["<redacted_token>"] = S["rq1_user_cause_words"] / S["user_turns"].replace(0, np.nan)

# 세션 사용 여부 기본 필터 후보: 빈 세션 제외 + use_in_analysis=True
base_mask = (~S["empty_session"]) & (S["use_in_analysis"])
S_base = S[base_mask].copy().reset_index(drop=True)

print("원본 세션 수:", len(S))
print("기본 필터 후 세션 수:", len(S_base))
display(S_base.head())

# ==== Cell 7 (code) ====
# 지속시간 분포 확인
q99 = S_base["duration_sec"].quantile(0.99)
q95 = S_base["duration_sec"].quantile(0.95)
print(f"duration_sec 95퍼센타일={q95:.1f}, 99퍼센타일={q99:.1f}")

# 상위 10개 긴 세션 점검
top_long = S_base.sort_values("duration_sec", ascending=False).head(10)
print("== 지속시간 상위 10 ==")
display(top_long[["session_file","condition","nickname","turns","duration_sec","duration_min","accepted","early_end_within2"]])

# (옵션) 윈저라이즈: 99퍼센타일 상한으로 절단한 보정 열 만들기
def winsorize(series, upper_q=0.99):
    cap = series.quantile(upper_q)
    return np.where(series > cap, cap, series)

S_base["duration_sec_wz"] = winsorize(S_base["duration_sec"], 0.99)
S_base["duration_min_wz"] = S_base["duration_sec_wz"] / 60.0

# 턴수도 극단치 확인
t_q99 = S_base["turns"].quantile(0.99)
print(f"turns 99퍼센타일={t_q99:.1f}")

# ==== Cell 8 (code) ====
def safe_mean(x):
    return np.nanmean(x) if len(x) else np.nan

def safe_median(x):
    return np.nanmedian(x) if len(x) else np.nan

agg = (S_base
       .groupby("condition")
       .agg(
           n_sessions = ("session_file","count"),
           n_users    = ("nickname","nunique"),
           mean_turns = ("turns", "mean"),
           median_turns = ("turns", "median"),
           mean_duration_min = ("duration_min", "mean"),
           median_duration_min = ("duration_min", "median"),
           mean_duration_min_wz = ("duration_min_wz", "mean"),
           accept_rate = ("accepted", "mean"),
           early_end_rate = ("early_end_within2", "mean"),
           mean_turns_to_accept = ("turns_to_accept", safe_mean),
           median_turns_to_accept = ("turns_to_accept", safe_median),
           rq1_qturns_mean = ("rq1_bot_question_turns", "mean"),
           rq1_qturn_ratio_mean = ("q_turn_ratio", "mean"),
           cause_words_mean = ("rq1_user_cause_words", "mean"),
           <redacted_token> = ("<redacted_token>", "mean"),
           sentiment_delta_mean = ("sentiment_delta","mean"),
       )
       .reset_index()
      )

print("== 세션 레벨 조건별 요약 ==")
display(agg.round(3))

# ==== Cell 9 (code) ====
import matplotlib.pyplot as plt

# 턴수 박스플롯
plt.figure()
S_base.boxplot(column="turns", by="condition")
plt.title("Session Turns by Condition"); plt.suptitle(""); plt.ylabel("Turns"); plt.xlabel("Condition")
plt.show()

# 지속시간(윈저라이즈) 박스플롯
plt.figure()
S_base.boxplot(column="duration_min_wz", by="condition")
plt.title("Session Duration (min, winsorized) by Condition"); plt.suptitle(""); plt.ylabel("Minutes"); plt.xlabel("Condition")
plt.show()

# 수락률 막대
acc = S_base.groupby("condition")["accepted"].mean().reset_index()
plt.figure()
plt.bar(acc["condition"], acc["accepted"])
plt.ylim(0,1); plt.title("Acceptance Rate by Condition"); plt.ylabel("Rate")
plt.show()

# 조기종료율 막대
ee = S_base.groupby("condition")["early_end_within2"].mean().reset_index()
plt.figure()
plt.bar(ee["condition"], ee["early_end_within2"])
plt.ylim(0,1); plt.title("Early End (≤2 turns after directive) by Condition"); plt.ylabel("Rate")
plt.show()

# RQ1: 원인 서술 단어수 박스플롯
plt.figure()
S_base.boxplot(column="rq1_user_cause_words", by="condition")
plt.title("Cause Words per Session by Condition"); plt.suptitle(""); plt.ylabel("Words"); plt.xlabel("Condition")
plt.show()

# ==== Cell 10 (code) ====
# RQ1: 질문-응답 탐색이 실제로 있었던 세션만
rq1_mask = S_base["rq1_bot_question_turns"] > 0
RQ1 = S_base[rq1_mask].copy()
rq1_agg = (RQ1.groupby("condition")
           .agg(
               n_sessions=("session_file","count"),
               qturns_mean=("rq1_bot_question_turns","mean"),
               cause_words_mean=("rq1_user_cause_words","mean"),
               qturn_ratio_mean=("q_turn_ratio","mean"),
               <redacted_token>=("<redacted_token>","mean"),
           )
           .reset_index())
print("== RQ1 유효 세션 요약 ==")
display(rq1_agg.round(3))

# RQ2: 지시가 있었던 세션만
rq2_mask = S_base["has_directive"]
RQ2 = S_base[rq2_mask].copy()
rq2_agg = (RQ2.groupby("condition")
           .agg(
               n_sessions=("session_file","count"),
               accept_rate=("accepted","mean"),
               mean_tta=("turns_to_accept", safe_mean),
               early_end_rate=("early_end_within2","mean"),
           )
           .reset_index())
print("== RQ2 유효 세션 요약 ==")
display(rq2_agg.round(3))

# RQ3: 기본 필터 그대로 사용
rq3_agg = (S_base.groupby("condition")
           .agg(
               n_sessions=("session_file","count"),
               mean_turns=("turns","mean"),
               median_turns=("turns","median"),
               mean_duration_min_wz=("duration_min_wz","mean"),
               early_end_rate=("early_end_within2","mean"),
           )
           .reset_index())
print("== RQ3 세션 요약 ==")
display(rq3_agg.round(3))

# ==== Cell 11 (code) ====
import numpy as np
import matplotlib.pyplot as plt

# 지시가 있었던 세션만 대상
T = S_base[S_base["has_directive"]].copy()

plt.figure()
for cond, dfc in T.groupby("condition"):
    # 수락한 세션들의 TTA
    tta = dfc["turns_to_accept"].dropna().astype(float).values
    # 수락 안 한 세션은 검열로 보고 '큰 값'으로 처리(여기선 turns+1로 설정)
    cens = dfc[dfc["turns_to_accept"].isna()]["turns"].astype(float).values + 1.0
    all_times = np.concatenate([tta, cens]) if len(cens) else tta

    if len(all_times)==0:
        continue
    xs = np.sort(all_times)
    # ECDF of "not yet accepted by t" ≈ 1 - (#accepted≤t / total)
    acc_counts = np.searchsorted(np.sort(tta), xs, side="right") if len(tta) else np.zeros_like(xs)
    surv = 1.0 - acc_counts / len(dfc)
    plt.step(xs, surv, where="post", label=cond)

plt.ylim(0,1.0)
plt.xlabel("Turns"); plt.ylabel("P(Not Accepted by t)")
plt.title("Pseudo-survival of Acceptance (by Turns)")
plt.legend()
plt.show()

# ==== Cell 12 (code) ====
# ==========================================
# 설문 닉네임 ↔ 로그 닉네임 매칭 (정규화 + 부분매칭)
# 붙일 위치: part_agg까지 만든 직후
# ==========================================
import re
import pandas as pd

# 1) 설문 파일 경로와 닉네임 컬럼명 지정
PRE_XLSX = r"data/cbt-i 사전 설문조사(응답).xlsx"  # ← 경로만 맞춰줘
NICK_COL = "마지막으로, 챗봇 대화 시 사용하실 닉네임을 작성해주세요.\n- 동일한 닉네임을 5일간 계속 사용해 주셔야 사용 로그 확인 후 사후 설문조사를 보내드릴 수 있습니다.\n- 대화 내용 분석도 작성해주신 닉네임을 기준으로 진행됩니다.\n(ex. 명륜, 성균, 킹고)"
ID_COL   = "1. 귀하의 성함이 어떻게 되십니까?"

# 2) 문자열 정규화 함수(대소문자/공백/문장부호/괄호/이모지 제거)
def norm_name(s: str) -> str:
    if not isinstance(s, str): 
        return ""
    s = s.strip().lower()
    s = re.sub(r"[\(\[].*?[\)\]]", "", s)   # 괄호/대괄호 안 내용 제거
    s = re.sub(r"\s+", "", s)               # 모든 공백 제거
    s = re.sub(r"[^\w가-힣]", "", s)        # 문자/숫자 외 제거
    return s

# 3) 설문에서 닉네임 테이블 추출
pre_raw = pd.read_excel(PRE_XLSX)
if NICK_COL not in pre_raw.columns:
    raise ValueError(f"[ERROR] 설문에 닉네임 문항 컬럼이 없습니다: {NICK_COL}")

nick_tbl = pre_raw[[ID_COL, NICK_COL]].rename(columns={NICK_COL:"survey_nickname"})
nick_tbl["survey_nickname"] = nick_tbl["survey_nickname"].astype(str).str.strip()
nick_tbl["nick_norm"] = nick_tbl["survey_nickname"].apply(norm_name)

# 4) 로그 참가자 집계(part_agg)에도 정규화 컬럼 부여
part = part_agg.copy()
part["nickname"] = part["nickname"].astype(str).str.strip()
part["nick_norm"] = part["nickname"].apply(norm_name)

# 5) 1차: 정규화 기준으로 정확 매칭
m1 = part.merge(nick_tbl, on="nick_norm", how="inner", suffixes=("_log","_survey"))
m1 = m1.drop_duplicates(subset=["nickname","survey_nickname"])

# 6) 2차(옵션): 여전히 남은 로그 닉네임에 대해 부분 포함 매칭 시도
unmatched_part = part[~part["nickname"].isin(m1["nickname"])]
maybe_matches = []
if not unmatched_part.empty:
    for _, r in unmatched_part.iterrows():
        nn = r["nick_norm"]
        # 길이가 너무 짧으면 오탑 가능성 ↑ → 길이≥3만 후보로
        if len(nn) < 3:
            continue
        cand = nick_tbl[
            nick_tbl["nick_norm"].str.contains(nn, na=False) |
            nick_tbl["nick_norm"].apply(lambda x: nn in x)
        ]
        for _, c in cand.iterrows():
            maybe_matches.append({
                "nickname": r["nickname"],
                "nick_norm_log": nn,
                "survey_nickname": c["survey_nickname"],
                "nick_norm_svy": c["nick_norm"],
                ID_COL: c[ID_COL],
            })
maybe_df = pd.DataFrame(maybe_matches).drop_duplicates()

# 7) 최종 매칭 테이블(우선 정확매칭, 필요 시 부분매칭 수동확인)
matched = m1.copy()
matched_cols = [
    "condition","nickname","survey_nickname", ID_COL, "n_sessions",
    "total_turns","mean_turns","accept_rate","mean_tta",
    "mean_qturns","mean_cause_words","early_end_rate","total_duration_s"
]
matched = matched[[c for c in matched_cols if c in matched.columns]]

# 8) 리포트 & 저장
print(f"[INFO] 로그-설문 매칭(정확): {matched['nickname'].nunique()}/{len(part['nickname'].unique())}명")
if not maybe_df.empty:
    print("[HINT] 일부 닉네임은 부분매칭 후보가 있습니다. 아래 CSV를 보고 수동 확인 후 머지하세요.")
    display(maybe_df.head(10))

# 저장
(Path(OUT)/"<redacted_token>.csv").write_text("", encoding="utf-8")  # 파일 초기화
matched.to_csv(Path(OUT, "<redacted_token>.csv"), index=False, encoding="utf-8-sig")
if not maybe_df.empty:
    maybe_df.to_csv(Path(OUT, "log_survey_maybe_fuzzy.csv"), index=False, encoding="utf-8-sig")
print("[OK] 저장:", Path(OUT, "<redacted_token>.csv"), 
      "(부분매칭 후보:", Path(OUT, "log_survey_maybe_fuzzy.csv"), ")")

# ==== Cell 13 (code) ====
# -*- coding: utf-8 -*-
# 로그×설문 매칭 + 변화량(Δ) vs 로그지표 상관/회귀/조절 분석 (자동 경로 탐색 포함)

import pandas as pd
import numpy as np
from pathlib import Path
import glob
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# =========================
# 0) 기본 설정
# =========================
# (필요시 바꿔도 됨) 실명 컬럼명
ID_COL = "1. 귀하의 성함이 어떻게 되십니까?"

# 로그지표 후보(파일에 없으면 자동 건너뜀)
BEHAV_VARS = [
    "n_sessions","total_turns","mean_turns","accept_rate","mean_tta",
    "mean_qturns","mean_cause_words","early_end_rate","total_duration_s"
]

# Δ-아웃컴 후보 접두사(*_delta 자동 탐지)
DELTA_PREFIXES = ["HBM_", "TPB_", "Stage_"]

# 공변량(있으면 사용)
COVS_BASE = ["ISI_sev_pre"]

# 출력 폴더
OUT_DIR = Path("analysis_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 1) 안전 로더(자동 경로 탐색)
# =========================
def smart_read_csv(preferred: Path, fallback_patterns):
    """
    preferred 경로가 있으면 사용.
    없으면 fallback_patterns(glob 패턴들)을 순서대로 검색해서
    가장 먼저 발견한 파일을 읽어옴.
    """
    if preferred and preferred.exists():
        print(f"[INFO] 사용 경로: {preferred}")
        return pd.read_csv(preferred), preferred

    for pat in fallback_patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            p = Path(hits[0])
            print(f"[INFO] 대체 경로 사용: {p} (패턴: {pat})")
            return pd.read_csv(p), p

    raise FileNotFoundError(
        f"CSV 파일을 찾을 수 없습니다.\n"
        f"- 기본 경로: {preferred}\n"
        f"- 시도한 패턴: {fallback_patterns}\n"
        f"경로를 확인하거나, 먼저 설문 파이프라인을 돌려 merged_scored.csv를 생성하세요."
    )

# 1) 로그×설문 매칭 CSV
LOG_SURVEY_MATCH_PATH = Path("data/analysis_out/<redacted_token>.csv")
logdf, LOG_SURVEY_MATCH_PATH = smart_read_csv(
    LOG_SURVEY_MATCH_PATH,
    fallback_patterns=[
        "analysis_out/<redacted_token>.csv",
        "data/**/<redacted_token>.csv",
        "**/<redacted_token>.csv",
    ],
)

# 2) 설문 스코어 병합 CSV
MERGED_SCORED_PATH = Path("data/exp_outputs/merged_scored.csv")
scored, MERGED_SCORED_PATH = smart_read_csv(
    MERGED_SCORED_PATH,
    fallback_patterns=[
        "exp_outputs/merged_scored.csv",
        "data/**/merged_scored.csv",
        "**/merged_scored.csv",
    ],
)

print(f"[OK] 로그 파일: {LOG_SURVEY_MATCH_PATH}")
print(f"[OK] 설문 파일: {MERGED_SCORED_PATH}")

# =========================
# 2) 병합 & 준비
# =========================
# 실명 컬럼 존재 확인
if ID_COL not in logdf.columns:
    raise ValueError(f"매칭파일에 '{ID_COL}' 컬럼이 없습니다. (현재 열: {logdf.columns.tolist()})")
if ID_COL not in scored.columns:
    raise ValueError(f"merged_scored.csv에 '{ID_COL}' 컬럼이 없습니다. (현재 열: {scored.columns.tolist()})")

# 병합 (실명 기준)
merged = logdf.merge(scored, on=ID_COL, how="inner", suffixes=("_log", "_svy"))
print(f"[INFO] 병합 완료: 로그 {len(logdf)}행, 설문 {len(scored)}행 → 교집합 {len(merged)}행")

# Δ-변화량 자동 탐지(*_delta)
delta_cols = [c for c in merged.columns if c.endswith("_delta") and any(c.startswith(p) for p in DELTA_PREFIXES)]
# 실제 존재하는 로그 변수만
behav_cols = [c for c in BEHAV_VARS if c in merged.columns]
# 실제 존재하는 공변량만
covs = [c for c in COVS_BASE if c in merged.columns]

print(f"[INFO] Δ-아웃컴 후보: {delta_cols}")
print(f"[INFO] 로그지표 사용: {behav_cols}")
print(f"[INFO] 공변량 사용: {covs}")

use = merged.copy()
# 숫자형 변환
for c in behav_cols + delta_cols + covs:
    use[c] = pd.to_numeric(use[c], errors="coerce")

# 범주 변수
if "group" in use.columns:
    use["group"] = use["group"].astype("category")
if "condition" in use.columns:
    use["condition"] = use["condition"].astype("category")

# =========================
# 3) 상관분석 (Pearson)
# =========================
corr_rows = []
for y in delta_cols:
    for x in behav_cols:
        sub = use[[x, y]].dropna()
        if len(sub) >= 5:
            r = sub[x].corr(sub[y])
            n = len(sub)
            if abs(r) < 1:
                t = r * np.sqrt((n-2)/(1-r**2))
                p = 2 * (1 - stats.t.cdf(abs(t), df=n-2))
            else:
                p = 0.0
            corr_rows.append({"outcome": y, "predictor": x, "r": r, "p": p, "N": n})

corr_df = pd.DataFrame(corr_rows).sort_values(["outcome","p"])
corr_df.to_csv(OUT_DIR/"corr_delta_vs_logs.csv", index=False, encoding="utf-8-sig")
print(f"[OK] 상관 결과 저장 → {OUT_DIR/'corr_delta_vs_logs.csv'}")

# =========================
# 4) 회귀분석 (OLS-HC3): Δ ~ 로그지표 + 공변량(+집단)
# =========================
def fit_ols_hc3(formula, data):
    try:
        m = smf.ols(formula, data=data.dropna()).fit(cov_type="HC3")
        return m
    except Exception:
        return None

reg_rows = []
for y in delta_cols:
    # 기본 예측자 세트(있으면 사용됨)
    base_predictors = [c for c in ["accept_rate","mean_turns","mean_cause_words","total_duration_s"] if c in behav_cols]
    cat = "group" if "group" in use.columns else ("condition" if "condition" in use.columns else None)

    parts = []
    if base_predictors: parts += base_predictors
    if covs: parts += covs
    if cat: parts += [f"C({cat})"]

    if not parts:
        continue

    formula = f"{y} ~ " + " + ".join(parts)
    m = fit_ols_hc3(formula, use)
    if m is None:
        continue

    for x in base_predictors:
        if x in m.params.index:
            reg_rows.append({
                "outcome": y,
                "predictor": x,
                "beta": float(m.params[x]),
                "p": float(m.pvalues[x]),
                "se_HC3": float(m.bse[x]),
                "R2": float(m.rsquared),
                "N": int(m.nobs),
                "model": formula
            })

reg_df = pd.DataFrame(reg_rows).sort_values(["outcome","p"])
reg_df.to_csv(OUT_DIR/"reg_delta_on_logs.csv", index=False, encoding="utf-8-sig")
print(f"[OK] 회귀 결과 저장 → {OUT_DIR/'reg_delta_on_logs.csv'}")

# =========================
# 5) 조절(상호작용): Δ ~ C(group/condition)*accept_rate + 공변량
# =========================
mod_rows = []
cat = "group" if "group" in use.columns else ("condition" if "condition" in use.columns else None)
if cat and "accept_rate" in behav_cols:
    for y in delta_cols:
        rhs = [f"C({cat})*accept_rate"] + covs
        formula = f"{y} ~ " + " + ".join(rhs)
        m = fit_ols_hc3(formula, use)
        if m is None: 
            continue
        ix_names = [ix for ix in m.params.index if ":" in ix]
        for ix in ix_names:
            mod_rows.append({
                "outcome": y,
                "interaction": ix,
                "beta": float(m.params[ix]),
                "p": float(m.pvalues[ix]),
                "se_HC3": float(m.bse[ix]),
                "R2": float(m.rsquared),
                "N": int(m.nobs),
                "model": formula
            })

mod_df = (pd.DataFrame(mod_rows)
          .sort_values(["outcome","p"])
          if len(mod_rows) else
          pd.DataFrame(columns=["outcome","interaction","beta","p","se_HC3","R2","N","model"]))
mod_df.to_csv(OUT_DIR/"<redacted_token>.csv", index=False, encoding="utf-8-sig")
print(f"[OK] 조절 결과 저장 → {OUT_DIR/'<redacted_token>.csv'}")

# =========================
# 6) FDR 보정(BH) – 회귀/조절 p값
# =========================
def fdr_bh_on_column(df, pcol="p"):
    if df is None or df.empty:
        return df
    rej, q, _, _ = multipletests(df[pcol].values, method="fdr_bh")
    out = df.copy()
    out["q_FDR_BH"] = q
    out["sig_FDR_05"] = rej
    return out

reg_df_fdr = fdr_bh_on_column(reg_df, "p")
mod_df_fdr = fdr_bh_on_column(mod_df, "p")

reg_df_fdr.to_csv(OUT_DIR/"reg_delta_on_logs_FDR.csv", index=False, encoding="utf-8-sig")
mod_df_fdr.to_csv(OUT_DIR/"<redacted_token>.csv", index=False, encoding="utf-8-sig")
print(f"[OK] FDR 보정 저장 → {OUT_DIR/'reg_delta_on_logs_FDR.csv'}, {OUT_DIR/'<redacted_token>.csv'}")

# =========================
# 7) 콘솔 요약
# =========================
print("\n=== [TOP 회귀 결과: Δ ~ 로그지표 + 공변량(+집단), p순] ===")
try:
    display(reg_df_fdr.head(15))
except Exception:
    print(reg_df_fdr.head(15).to_string(index=False))

print("\n=== [상관: Δ vs 로그지표, p순] ===")
try:
    display(corr_df.head(15))
except Exception:
    print(corr_df.head(15).to_string(index=False))

if not mod_df_fdr.empty:
    print("\n=== [조절: Δ ~ C(group/condition)*accept_rate + 공변량, p순] ===")
    try:
        display(mod_df_fdr.head(15))
    except Exception:
        print(mod_df_fdr.head(15).to_string(index=False))
else:
    print("\n[참고] 조절분석: group/condition 또는 accept_rate가 없어 스킵됐거나 유효 표본 부족.")
