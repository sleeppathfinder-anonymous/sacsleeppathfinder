# -*- coding: utf-8 -*-
# Generated from notebook on 2025-10-18T03:56:19
# Cells converted from Jupyter Notebook. Markdown cells are included as comments.

from __future__ import annotations


# ==== Cell 1 (code) ====
# -*- coding: utf-8 -*-
"""
CBT-I 실험(30명) 1차 분석 파이프라인 — Jupyter/Colab 버전
(PAQ-S 6문항 확정 + 느슨한 헤더 매칭 + PAQ-변화량 상관/회귀/조절 추가)

- 입력: 사전/사후 설문 xlsx
- 처리: 문항 점수화(1~5), 하위척도 산출(역문항/방향성 통일), 병합
- 통계: 혼합효과모형(실패 시 GEE 폴백), Δ-검정/ANCOVA, Stage 전이(카이제곱/Fisher, 순서형 로지스틱),
        UEQ-S/SDT 집단차(t/ANCOVA), PAQ↔Δ(HBM/TPB) 상관·회귀·조절, FDR(BH)
- 산출: merged_scored.csv, reliability.json, stats_report.txt
"""

import json, warnings, re
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

try:
    import pingouin as pg
    HAS_PINGOUIN = True
except Exception:
    HAS_PINGOUIN = False

from statsmodels.genmod.<redacted_token> import GEE
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.cov_struct import Exchangeable

# =========================
# 0) 기본 설정
# =========================
EXP_NAMES  = set(["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o"])
CTRL_NAMES = set(["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O"])

ID_COL_DEFAULT = "1. 귀하의 성함이 어떻게 되십니까?"

# =========================
# 1) 점수 매핑(한국어 → 1~5)
# =========================
LIKERT_5 = {
    # 동의/빈도형
    "매우 그렇다":5, "그렇다":4, "보통이다":3, "그렇지 않다":2,
    "매우 그렇지 않다":1, "전혀 그렇지 않다":1,
    # ISI 심각도형
    "매우 심함":5, "심함":4, "중간":3, "약간 있음":2, "전혀 없음":1,
    # 만족도형
    "매우 불만족":1, "불만족":2, "중간":3, "보통":3, "만족":4, "매우 만족":5,
    # 기능방해형
    "매우 많이 방해됨":5, "많이 방해됨":4, "다소 방해됨":3, "조금 방해됨":2, "전혀 방해되지 않음":1,
    # 사회적 인식(눈에 띔)형
    "전혀 눈에 띄지 않음":1, "약간 눈에 띔":2, "조금 눈에 띔":2, "다소 눈에 띔":3, "상당히 눈에 띔":4, "매우 눈에 띔":5,
}

STAGE_5 = {
    "생각 없음 (변화할 계획이 전혀 없음)":1,
    "생각 중 (변화 필요성을 인식하지만 아직 실행하지 않음)":2,
    "준비 (가까운 시일 내 실행할 준비 중)":3,
    "실행 (현재 변화를 실행하고 있음)":4,
    "유지 (변화를 지속적으로 유지하고 있음)":5,
}

# =========================
# 2) 컬럼 접두사(느슨한 매칭)
# =========================
PRE_ISI = [
    "1. 잠들기 어려움", "2. 자주 깨는 등 잠을 유지", "3. 너무 일찍 깨서 다시 잠들기 어려움",
    "4. 현재의 수면 양상에 대해 얼마나 만족", "5. 수면 문제 때문에 낮 동안의 일상생활",
    "6. 수면 문제로 인해 다른 사람들이", "7. 현재의 수면 문제로 인해 스스로 느끼는 걱정"
]

PRE_HBM = [
    "1. 수면 문제는 내 건강", "2. 수면 부족은 내 기분", "3. 지금의 수면 문제를 방치하면",
    "4. 현재 내 수면 패턴은", "5. 나는 수면 문제 위험군", "6. 최근 내 수면은 건강하지",
    "7. 나는 내 수면 문제가 왜 발생", "8. 나는 내 생활 습관 중", "9. 내 불면의 근본적인 원인"
]

PRE_TPB = [
    "1. 수면 개선 행동은 내게 도움이", "2. 수면 개선 행동은 내 삶의 질",
    "3. 수면 습관 실천은 수면 문제를 줄이는",
    "4. 부모/형제자매는 내가 수면 개선 행동을 하길",
    "5. 친한 친구들은 내가 수면 개선 행동을 하길",
    "6. 방해 요인이 있어도 나는 수면 개선 행동을 해낼",
    "7. 하루가 바빠도 나는 수면 개선 행동을 유지",
    "8. 피곤하거나 스트레스를 받아도 나는 수면 개선 행동을 지속",
    "9. 나는 다음 1주일 동안 수면 개선 행동을 꾸준히 유지할",
    "10. 나는 수면 개선 행동을 구체적으로 계획하고 실행할"
]
PRE_STAGE = "11. 수면 개선 행동에 대한 현재 나의 상태"

# PAQ-S 6문항
PRE_PAQ6 = [
    "내가 기분이 나쁠 때(불쾌한 감정을 느낄 때), 그 감정을 설명할 적절한 말을 찾을 수 없다",
    "내가 기분이 나쁠 때, 내가 슬픈 건지, 화난 건지, 두려운 건지 구분할 수 없다",
    "나는 내가 느끼는 감정을 무시하는 경향이 있다",
    "내가 기분이 좋을 때(유쾌한 감정을 느낄 때), 그 감정을 설명할 적절한 말을 찾을 수 없다",
    "내가 기분이 좋을 때, 내가 행복한 건지, 신난 건지, 즐거운 건지 구분할 수 없다",
    "나는 내 감정에 주의를 기울이지 않는다"
]

POST_HBM  = PRE_HBM
POST_TPB  = PRE_TPB
POST_STAGE= PRE_STAGE

POST_SDT = [
    "1. 나는 챗봇이 내게 선택과 대안을 제공", "2. 나는 챗봇에게서 이해받고 있다고",
    "3. 챗봇은 내가 질문하도록 격려", "4. 나는 챗봇에게 많은 신뢰를",
    "5. 챗봇은 내 의견과 관점을 반영", "6. 챗봇은 내 감정을 존중"
]
POST_UEQS = [f"본인의 느낌에 가장 가까운 위치에 표시해 주세요. ({k}/8)" for k in range(1, 9)]

UEQ_P_IDX = [0,1,2,3]
UEQ_H_IDX = [4,5,6,7]

HBM_SEV_IDX = [0,1,2]
HBM_SUS_IDX = [3,4,5]
HBM_AWR_IDX = [6,7,8]

TPB_BB   = [0,1,2]
TPB_SN   = [3,4]
TPB_SE   = [5,6,7]
TPB_INT  = [8,9]

# =========================
# 3) 헬퍼 (느슨한 매칭)
# =========================
def _normalize(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^\w가-힣]", "", s)
    return s

def find_cols(df: pd.DataFrame, prefixes: List[str]) -> List[str]:
    cols = []
    norm_cols = {c: _normalize(c) for c in df.columns}
    for p in prefixes:
        npfx = _normalize(p)
        hit = [c for c, nc in norm_cols.items() if nc.startswith(npfx)]
        if not hit:
            hit = [c for c, nc in norm_cols.items() if npfx in nc]
        if hit: cols.append(hit[0])
        else: print(f"[WARN] 컬럼 누락(느슨매칭 실패): {p}")
    return cols

def cronbach_alpha(df_num: pd.DataFrame) -> float:
    x = df_num.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if x.shape[1] < 2: return float("nan")
    item_vars = x.var(axis=0, ddof=1)
    total_var = x.sum(axis=1).var(ddof=1)
    k = x.shape[1]
    if total_var == 0 or k <= 1: return float("nan")
    return float((k/(k-1))*(1-(item_vars.sum()/total_var)))

def _rev(v):
    v = pd.to_numeric(v, errors="coerce")
    return 6 - v if pd.notna(v) else v

def score_block(df: pd.DataFrame, cols: List[str], mapping: Dict[str,int], reverse_idx: List[int]=None):
    sub = df[cols].replace(mapping)
    sub = sub.apply(pd.to_numeric, errors="coerce")
    if reverse_idx:
        for i in reverse_idx:
            if i < sub.shape[1]:
                sub.iloc[:, i] = sub.iloc[:, i].apply(_rev)
    alpha = cronbach_alpha(sub)
    score = sub.mean(axis=1, skipna=True)
    return score, alpha, sub

def audit_unmapped(df: pd.DataFrame, cols: List[str], mapping: Dict[str,int]) -> Dict[str, List[str]]:
    bad = {}
    for c in cols:
        vals = pd.Series(df[c].unique()).dropna().astype(str)
        unm = [v for v in vals if (v not in mapping) and (not v.replace('.', '', 1).isdigit())]
        if unm: bad[c] = unm
    return bad

def hedges_g(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return np.nan
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    sp_num = (nx-1)*sx2 + (ny-1)*sy2
    sp_den = (nx+ny-2)
    if sp_den <= 0: return np.nan
    sp = np.sqrt(sp_num / sp_den)
    if sp == 0: return np.nan
    d = (np.mean(x)-np.mean(y))/sp
    J = 1 - (3/(4*(nx+ny)-9))
    return d*J

# =========================
# 4) 로드 & 라벨링
# =========================
def load_and_label(pre_path, post_path, id_col):
    pre  = pd.read_excel(pre_path)
    post = pd.read_excel(post_path)
    assert id_col in pre.columns and id_col in post.columns, f"{id_col} 컬럼 확인 필요"
    pre[id_col]  = pre[id_col].astype(str).str.strip()
    post[id_col] = post[id_col].astype(str).str.strip()

    def lab(n):
        if n in EXP_NAMES: return "exp"
        if n in CTRL_NAMES: return "ctrl"
        return "unknown"
    pre["group"]  = pre[id_col].apply(lab)
    post["group"] = post[id_col].apply(lab)
    return pre, post

# =========================
# 5) 스코어링(사전/사후)
# =========================
def score_pre(df, id_col):
    out = df[[id_col,"group"]].copy()
    info = {}

    # ISI
    isi_cols = find_cols(df, PRE_ISI)
    isi_score, isi_alpha, isi_sub = score_block(df, isi_cols, LIKERT_5, reverse_idx=[3])
    out["ISI_sev_pre"] = isi_score
    info["ISI_alpha_pre"] = isi_alpha
    if len(isi_cols) >= 4:
        out["ISI_sat_pre"] = df[isi_cols[3]].replace(LIKERT_5).apply(pd.to_numeric, errors="coerce")

    # HBM
    hbm_cols = find_cols(df, PRE_HBM)
    _, _, hbm_sub = score_block(df, hbm_cols, LIKERT_5)
    out["HBM_SEV_pre"] = hbm_sub.iloc[:, HBM_SEV_IDX].mean(axis=1, skipna=True)
    out["HBM_SUS_pre"] = hbm_sub.iloc[:, HBM_SUS_IDX].mean(axis=1, skipna=True)
    out["HBM_AWR_pre"] = hbm_sub.iloc[:, HBM_AWR_IDX].mean(axis=1, skipna=True)
    info["HBM_SEV_alpha_pre"] = cronbach_alpha(hbm_sub.iloc[:, HBM_SEV_IDX])
    info["HBM_SUS_alpha_pre"] = cronbach_alpha(hbm_sub.iloc[:, HBM_SUS_IDX])
    info["HBM_AWR_alpha_pre"] = cronbach_alpha(hbm_sub.iloc[:, HBM_AWR_IDX])

    # TPB
    tpb_cols = find_cols(df, PRE_TPB)
    _, _, tpb_sub = score_block(df, tpb_cols, LIKERT_5)
    out["TPB_BB_pre"]  = tpb_sub.iloc[:, TPB_BB].mean(axis=1, skipna=True)
    out["TPB_SN_pre"]  = tpb_sub.iloc[:, TPB_SN].mean(axis=1, skipna=True)
    out["TPB_SE_pre"]  = tpb_sub.iloc[:, TPB_SE].mean(axis=1, skipna=True)
    out["TPB_INT_pre"] = tpb_sub.iloc[:, TPB_INT].mean(axis=1, skipna=True)
    info["TPB_BB_alpha_pre"]  = cronbach_alpha(tpb_sub.iloc[:, TPB_BB])
    info["TPB_SN_alpha_pre"]  = cronbach_alpha(tpb_sub.iloc[:, TPB_SN])
    info["TPB_SE_alpha_pre"]  = cronbach_alpha(tpb_sub.iloc[:, TPB_SE])
    info["TPB_INT_alpha_pre"] = cronbach_alpha(tpb_sub.iloc[:, TPB_INT])

    # Stage
    stage_col = find_cols(df, [PRE_STAGE])
    out["Stage_pre"] = df[stage_col[0]].map(STAGE_5) if stage_col else np.nan

    # PAQ-S (6문항)
    paq_cols = find_cols(df, PRE_PAQ6)
    paq_score, paq_alpha, paq_sub = score_block(df, paq_cols, LIKERT_5)
    out["PAQ_pre"]       = paq_score
    out["PAQ_DDF_neg"]   = paq_sub.iloc[:,0]
    out["PAQ_DIF_neg"]   = paq_sub.iloc[:,1]
    out["PAQ_EOT_neg"]   = paq_sub.iloc[:,2]
    out["PAQ_DDF_pos"]   = paq_sub.iloc[:,3]
    out["PAQ_DIF_pos"]   = paq_sub.iloc[:,4]
    out["PAQ_EOT_pos"]   = paq_sub.iloc[:,5]
    out["PAQ_total_pre"] = paq_sub.mean(axis=1, skipna=True)
    info["PAQ_alpha_pre"]    = paq_alpha
    info["PAQ_matched_pre"]  = paq_cols
    info["PAQ_unmapped_pre"] = audit_unmapped(df, paq_cols, LIKERT_5)

    # 매핑 로그
    info["matched_pre"] = {"ISI":isi_cols,"HBM":hbm_cols,"TPB":tpb_cols,"Stage":stage_col}
    info["unmapped_pre"] = {
        "ISI": audit_unmapped(df, isi_cols, LIKERT_5),
        "HBM": audit_unmapped(df, hbm_cols, LIKERT_5),
        "TPB": audit_unmapped(df, tpb_cols, LIKERT_5),
    }
    return out, info

def score_post(df, id_col):
    out = df[[id_col,"group"]].copy()
    info = {}

    # HBM
    hbm_cols = find_cols(df, POST_HBM)
    _, _, hbm_sub = score_block(df, hbm_cols, LIKERT_5)
    out["HBM_SEV_post"] = hbm_sub.iloc[:, HBM_SEV_IDX].mean(axis=1, skipna=True)
    out["HBM_SUS_post"] = hbm_sub.iloc[:, HBM_SUS_IDX].mean(axis=1, skipna=True)
    out["HBM_AWR_post"] = hbm_sub.iloc[:, HBM_AWR_IDX].mean(axis=1, skipna=True)
    info["HBM_SEV_alpha_post"] = cronbach_alpha(hbm_sub.iloc[:, HBM_SEV_IDX])
    info["HBM_SUS_alpha_post"] = cronbach_alpha(hbm_sub.iloc[:, HBM_SUS_IDX])
    info["HBM_AWR_alpha_post"] = cronbach_alpha(hbm_sub.iloc[:, HBM_AWR_IDX])

    # TPB
    tpb_cols = find_cols(df, POST_TPB)
    _, _, tpb_sub = score_block(df, tpb_cols, LIKERT_5)
    out["TPB_BB_post"]  = tpb_sub.iloc[:, TPB_BB].mean(axis=1, skipna=True)
    out["TPB_SN_post"]  = tpb_sub.iloc[:, TPB_SN].mean(axis=1, skipna=True)
    out["TPB_SE_post"]  = tpb_sub.iloc[:, TPB_SE].mean(axis=1, skipna=True)
    out["TPB_INT_post"] = tpb_sub.iloc[:, TPB_INT].mean(axis=1, skipna=True)
    info["TPB_BB_alpha_post"]  = cronbach_alpha(tpb_sub.iloc[:, TPB_BB])
    info["TPB_SN_alpha_post"]  = cronbach_alpha(tpb_sub.iloc[:, TPB_SN])
    info["TPB_SE_alpha_post"]  = cronbach_alpha(tpb_sub.iloc[:, TPB_SE])
    info["TPB_INT_alpha_post"] = cronbach_alpha(tpb_sub.iloc[:, TPB_INT])

    # Stage
    stage_col = find_cols(df, [POST_STAGE])
    out["Stage_post"] = df[stage_col[0]].map(STAGE_5) if stage_col else np.nan

    # SDT + UEQ-S
    sdt_cols = find_cols(df, POST_SDT)
    sdt_score, sdt_alpha, _ = score_block(df, sdt_cols, LIKERT_5)
    out["SDT_post"] = sdt_score
    info["SDT_alpha_post"] = sdt_alpha

    ueq_cols = find_cols(df, POST_UEQS)
    ueq_raw  = df[ueq_cols].apply(pd.to_numeric, errors="coerce")
    out["UEQS_P_post"] = ueq_raw.iloc[:, UEQ_P_IDX].mean(axis=1, skipna=True)
    out["UEQS_H_post"] = ueq_raw.iloc[:, UEQ_H_IDX].mean(axis=1, skipna=True)
    out["UEQS_post"]   = ueq_raw.mean(axis=1, skipna=True)
    info["UEQS_items"] = ueq_cols

    info["matched_post"] = {"HBM":hbm_cols,"TPB":tpb_cols,"Stage":stage_col,"SDT":sdt_cols,"UEQ-S":ueq_cols}
    info["unmapped_post"] = {
        "HBM": audit_unmapped(df, hbm_cols, LIKERT_5),
        "TPB": audit_unmapped(df, tpb_cols, LIKERT_5),
        "SDT": audit_unmapped(df, sdt_cols, LIKERT_5),
    }
    return out, info

# =========================
# 6) 통계 (혼합모형 + 폴백)
# =========================
def _confint_from_result(res, param_name, z=1.96):
    try:
        beta = res.params[param_name]
        se   = res.bse[param_name]
        p    = res.pvalues[param_name]
        lo, hi = beta - z*se, beta + z*se
        return float(beta), float(se), float(lo), float(hi), float(p)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan

def fit_mixed_or_gee(long_df, id_col, dv, covariates):
    long = long_df.copy()
    long["group"] = long["group"].astype("category")
    long["time"]  = long["time"].astype("category")

    rhs_terms = ["C(group)", "C(time)", "C(group):C(time)"] + (covariates or [])
    rhs = " + ".join(rhs_terms)
    formula = f"{dv} ~ {rhs}"
    target_param = "C(group)[T.exp]:C(time)[T.pre]"

    try:
        md = smf.mixedlm(formula, long, groups=long[id_col])
        m  = md.fit(reml=False, method="lbfgs")
        beta, se, lo, hi, p = _confint_from_result(m, target_param)
        return {"engine":"MixedLM", "summary":str(m.summary()),
                "param":target_param, "beta":beta, "se":se, "ci":(lo,hi), "p":p}
    except Exception:
        pass
    try:
        md = smf.mixedlm(formula, long, groups=long[id_col])
        m  = md.fit(reml=False, method="powell", maxiter=200, disp=False)
        beta, se, lo, hi, p = _confint_from_result(m, target_param)
        return {"engine":"MixedLM(powell)", "summary":str(m.summary()),
                "param":target_param, "beta":beta, "se":se, "ci":(lo,hi), "p":p}
    except Exception:
        pass
    try:
        gee = GEE.from_formula(formula, groups=id_col, data=long,
                               family=Gaussian(), cov_struct=Exchangeable())
        gres = gee.fit()
        beta, se, lo, hi, p = _confint_from_result(gres, target_param)
        return {"engine":"GEE(exchangeable, robust)", "summary":str(gres.summary()),
                "param":target_param, "beta":beta, "se":se, "ci":(lo,hi), "p":p}
    except Exception as e3:
        return {"engine":"FAILED", "error":f"{e3}", "param":target_param,
                "beta":np.nan, "se":np.nan, "ci":(np.nan, np.nan), "p":np.nan}

def mixed_model_block(merged: pd.DataFrame, id_col: str, dv_list: List[str], covariates: List[str]=None) -> str:
    lines = ["\n[혼합효과모형]\n"]
    covariates = [c for c in (covariates or []) if c in merged.columns]
    for base in dv_list:
        pre, post = f"{base}_pre", f"{base}_post"
        if pre not in merged.columns or post not in merged.columns:
            lines.append(f"- {base}: 컬럼 부족\n"); continue

        keep = [id_col, "group", pre, post] + covariates
        dat  = merged[keep].dropna(subset=["group", pre, post]).copy()
        long = pd.melt(
            dat, id_vars=[id_col,"group"]+covariates,
            value_vars=[pre, post], var_name="time", value_name=base
        )
        long["time"] = long["time"].map({pre:"pre", post:"post"})

        res = fit_mixed_or_gee(long, id_col=id_col, dv=base, covariates=covariates)
        if res.get("engine") == "FAILED":
            lines.append(f"- {base} model FAILED: {res.get('error')}\n")
        else:
            beta, se, (lo,hi), p = res["beta"], res["se"], res["ci"], res["p"]
            lines.append(
                f"- {base} {res['engine']} | G×T: β={beta:.3f}, SE={se:.3f}, 95% CI [{lo:.3f}, {hi:.3f}], p={p:.3f}\n"
            )

        # 보조: Δ-검정
        tmp = dat.copy()
        tmp[f"{base}_delta"] = tmp[post] - tmp[pre]
        g_exp = tmp.loc[tmp["group"]=="exp", f"{base}_delta"]
        g_ctrl= tmp.loc[tmp["group"]=="ctrl",f"{base}_delta"]
        if len(g_exp)>=3 and len(g_ctrl)>=3:
            t, p = stats.ttest_ind(g_exp, g_ctrl, equal_var=False)
            g = hedges_g(g_exp, g_ctrl)
            lines.append(f"  Δ(사후-사전) Welch t: t≈{t:.3f}, p={p:.4f}, Hedges g={g:.3f}, Δ_exp={g_exp.mean():.2f}, Δ_ctrl={g_ctrl.mean():.2f}\n")

    return "".join(lines)

def ancova_delta_blocks(merged: pd.DataFrame, blocks: List[Tuple[str,str,str]]):
    lines = ["\n[Δ-검정 & ANCOVA]\n"]
    for pre, post, label in blocks:
        if pre not in merged.columns or post not in merged.columns:
            continue
        s = merged.dropna(subset=[pre, post, "group"]).copy()
        s["delta"] = s[post] - s[pre]
        g1 = s.loc[s["group"]=="exp", "delta"]
        g2 = s.loc[s["group"]=="ctrl","delta"]
        if len(g1)>=3 and len(g2)>=3:
            t, p = stats.ttest_ind(g1, g2, equal_var=False)
            g = hedges_g(g1, g2)
            lines.append(f"- {label} Δ Welch t: t≈{t:.3f}, p={p:.4f}, g={g:.3f}, Δ_exp={g1.mean():.2f}, Δ_ctrl={g2.mean():.2f}\n")
        try:
            mod = smf.ols(f"{post} ~ C(group) + {pre}", data=s).fit()
            aov = sm.stats.anova_lm(mod, typ=2)
            if "C(group)" in aov.index:
                ss_eff = aov.loc["C(group)","sum_sq"]; ss_err = aov.loc["Residual","sum_sq"]
                pes = ss_eff/(ss_eff+ss_err) if (ss_eff+ss_err)>0 else np.nan
                lines.append(f"  ANCOVA {label}: F={aov.loc['C(group)','F']:.3f}, p={aov.loc['C(group)','PR(>F)']:.4f}, partial eta²={pes:.3f}\n")
        except Exception as e:
            lines.append(f"  ANCOVA {label} 오류: {e}\n")
    return "".join(lines)

# ---------- NEW: PAQ ↔ Δ(HBM/TPB) 상관/회귀/조절 ----------
def paq_delta_association(merged: pd.DataFrame) -> str:
    """
    1) 집단 무관 상관: PAQ_total_pre(및 하위척도) ~ Δ(DVs)
    2) 기저 통제 회귀: Δ ~ PAQ_total_pre + ISI_sev_pre  (HC3)
    3) 조절효과: Δ ~ C(group) * PAQ_total_pre + ISI_sev_pre  (HC3)
    """
    lines = ["\n[PAQ-S × 변화량 (상관/회귀/조절)]\n"]
    dvs = ["HBM_SEV","HBM_SUS","HBM_AWR","TPB_BB","TPB_SN","TPB_SE","TPB_INT"]
    paq_vars = ["PAQ_total_pre","PAQ_DIF_neg","PAQ_DDF_neg","PAQ_EOT_neg","PAQ_DIF_pos","PAQ_DDF_pos","PAQ_EOT_pos","PAQ_pre"]
    paq_vars = [v for v in paq_vars if v in merged.columns]

    # 1) 상관 (집단 무관)
    lines.append("- (상관) 집단 무관 Pearson r\n")
    for dv in dvs:
        dcol = f"{dv}_delta"
        if dcol not in merged.columns: 
            continue
        s = merged.dropna(subset=[dcol] + paq_vars).copy()
        if len(s) < 6: 
            lines.append(f"  · {dv}: 샘플 부족\n"); 
            continue
        # 핵심: PAQ_total_pre 우선
        if "PAQ_total_pre" in s.columns:
            r,p = stats.pearsonr(s["PAQ_total_pre"], s[dcol])
            lines.append(f"  · {dv} ~ PAQ_total_pre: r={r:.3f}, p={p:.4f}, N={len(s)}")
        # 하위척도도 간단히 보고
        for pv in [x for x in paq_vars if x!="PAQ_total_pre"]:
            r,p = stats.pearsonr(s[pv], s[dcol])
            lines.append(f"    - {dv} ~ {pv}: r={r:.3f}, p={p:.4f}")
        lines.append("\n")

    # 2) 회귀 (Δ ~ PAQ + ISI)
    lines.append("- (회귀) Δ ~ PAQ_total_pre + ISI_sev_pre  [HC3]\n")
    for dv in dvs:
        dcol = f"{dv}_delta"
        if dcol not in merged.columns: 
            continue
        keep = [dcol, "PAQ_total_pre", "ISI_sev_pre"]
        s = merged.dropna(subset=keep).copy()
        if len(s) < 8: 
            lines.append(f"  · {dv}: 샘플 부족\n"); 
            continue
        try:
            fit = smf.ols(f"{dcol} ~ PAQ_total_pre + ISI_sev_pre", data=s).fit(cov_type="HC3")
            b = fit.params.get("PAQ_total_pre", np.nan)
            p = fit.pvalues.get("PAQ_total_pre", np.nan)
            r2= fit.rsquared
            lines.append(f"  · {dv}: β(PAQ)={b:.3f}, p(PAQ)={p:.4f}, R²={r2:.3f}, N={len(s)}")
        except Exception as e:
            lines.append(f"  · {dv}: 회귀 오류 {e}")
    lines.append("\n")

    # 3) 조절효과 (Δ ~ group * PAQ + ISI)
    lines.append("- (조절) Δ ~ C(group) * PAQ_total_pre + ISI_sev_pre  [HC3]\n")
    if "PAQ_total_pre" not in merged.columns:
        lines.append("  · PAQ_total_pre 없음 → 생략\n"); 
        return "".join(lines)
    for dv in dvs:
        dcol = f"{dv}_delta"
        if dcol not in merged.columns: 
            continue
        keep = [dcol, "group", "PAQ_total_pre", "ISI_sev_pre"]
        s = merged.dropna(subset=keep).copy()
        if s["group"].nunique()!=2 or (s.groupby("group").size()<3).any() or len(s)<10:
            lines.append(f"  · {dv}: 샘플/집단 불충분\n"); 
            continue
        try:
            fit = smf.ols(f"{dcol} ~ C(group) * PAQ_total_pre + ISI_sev_pre", data=s).fit(cov_type="HC3")
            term = "C(group)[T.exp]:PAQ_total_pre"
            b = fit.params.get(term, np.nan)
            p = fit.pvalues.get(term, np.nan)
            lines.append(f"  · {dv}: β(상호작용)={b:.3f}, p={p:.4f}, N={len(s)}")
        except Exception as e:
            lines.append(f"  · {dv}: 조절회귀 오류 {e}")
    lines.append("\n")
    return "".join(lines)

# ---------- Stage & UEQ-S 기존 블록 ----------
def stage_transition_tests(merged: pd.DataFrame, id_col: str, debug=False) -> str:
    lines = ["\n[Stage 전이 분석]\n"]
    need_cols = ["group","Stage_pre","Stage_post"]
    if any(c not in merged.columns for c in need_cols):
        lines.append("- Stage 컬럼 없음\n"); return "".join(lines)

    df = merged.dropna(subset=need_cols).copy()
    df["entered_ready_or_more"] = (df["Stage_post"]>=3).astype(int)
    tab = pd.crosstab(df["group"], df["entered_ready_or_more"])
    lines.append(f"- 진입(>=준비) 교차표:\n{tab}\n")
    try:
        if tab.shape==(2,2):
            oddsratio, p = stats.fisher_exact(tab)
            lines.append(f"- Fisher's exact: OR={oddsratio:.3f}, p={p:.4f}\n")
        else:
            chi2, p, dof, exp = stats.chi2_contingency(tab)
            lines.append(f"- Chi-square: χ2({dof})={chi2:.3f}, p={p:.4f}\n")
    except Exception as e:
        lines.append(f"- Stage 이항검정 오류: {e}\n")

    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        df_ord = df.dropna(subset=["Stage_pre","Stage_post"]).copy()
        X = pd.get_dummies(df_ord[["group","Stage_pre"]], drop_first=True).astype(float)

        if "ISI_sev_pre" in merged.columns:
            isi_aligned = df_ord[[id_col]].merge(
                merged[[id_col,"ISI_sev_pre"]], on=id_col, how="left"
            )["ISI_sev_pre"].astype(float)
            X = X.join(isi_aligned.rename("ISI_sev_pre"))

        y = df_ord["Stage_post"].astype(int)
        mod = OrderedModel(y, X, distr='logit')
        res = mod.fit(method='bfgs', disp=False)
        if debug:
            lines.append(f"- 순서형 로지스틱(요약):\n{res.summary()}\n")

        try:
            ci = res.conf_int()
            params = res.params
            keys = [k for k in params.index if k in ["Stage_pre","group_exp","ISI_sev_pre"]]
            lines.append("- 순서형 로지스틱(OR 보고):\n")
            for k in keys:
                beta = params[k]; lo, hi = ci.loc[k, 0], ci.loc[k, 1]
                OR, OR_lo, OR_hi = np.exp(beta), np.exp(lo), np.exp(hi)
                pval = res.pvalues[k] if hasattr(res, "pvalues") else np.nan
                lines.append(f"  · {k}: OR={OR:.2f}, 95% CI [{OR_lo:.2f}, {OR_hi:.2f}], p={pval:.4f}\n")
        except Exception as e:
            lines.append(f"- OR 변환 출력 오류: {e}\n")

    except Exception as e:
        lines.append(f"- 순서형 로지스틱 오류: {e}\n")
    return "".join(lines)

def ueq_s_analysis(merged: pd.DataFrame) -> str:
    lines = ["\n[UEQ-S/SDT 사용자 경험]\n"]
    candidate_covs = [
        "ISI_sev_pre","HBM_SEV_pre","HBM_SUS_pre","TPB_INT_pre",
        "PAQ_total_pre","PAQ_DIF_neg","PAQ_DDF_neg","PAQ_EOT_neg",
        "PAQ_DIF_pos","PAQ_DDF_pos","PAQ_EOT_pos",
        "time_on_task","turns"
    ]
    covs = [c for c in candidate_covs if c in merged.columns]

    def _ttest_and_ancova(col):
        out = []
        if col not in merged.columns:
            out.append(f"- {col}: 컬럼 없음"); 
            return "\n".join(out)

        s = merged.dropna(subset=[col,"group"])
        if s["group"].nunique()==2 and all(s.groupby("group").size()>=3):
            g_exp = s.loc[s["group"]=="exp", col]
            g_ctrl= s.loc[s["group"]=="ctrl",col]
            t, p = stats.ttest_ind(g_exp, g_ctrl, equal_var=False)
            g = hedges_g(g_exp, g_ctrl)
            out.append(f"- {col} 독립 t: t≈{t:.3f}, p={p:.4f}, Hedges g={g:.3f}, M_exp={g_exp.mean():.2f}, M_ctrl={g_ctrl.mean():.2f}")
        else:
            out.append(f"- {col} 독립 t: 샘플 부족")

        if covs:
            try:
                formula = f"{col} ~ C(group) + " + " + ".join(covs)
                mod = smf.ols(formula, data=merged.dropna(subset=[col,"group"]+covs)).fit()
                aov = sm.stats.anova_lm(mod, typ=2)
                if "C(group)" in aov.index:
                    ss_eff=aov.loc["C(group)","sum_sq"]; ss_err=aov.loc["Residual","sum_sq"]
                    pes = ss_eff/(ss_eff+ss_err) if (ss_eff+ss_err)>0 else np.nan
                    out.append(f"- {col} ANCOVA(covs={covs}): F={aov.loc['C(group)','F']:.3f}, p={aov.loc['C(group)','PR(>F)']:.4f}, partial eta²={pes:.3f}")
            except Exception as e:
                out.append(f"- {col} ANCOVA 오류: {e}")
        else:
            out.append(f"- {col} ANCOVA: 사용 가능한 공변량 없음")

        return "\n".join(out)

    for col in ["UEQS_post","UEQS_P_post","UEQS_H_post", "SDT_post"]:
        lines.append(_ttest_and_ancova(col))

    return "\n".join(lines)

def multiple_testing_fdr(pvals: Dict[str,float], alpha=0.05):
    keys = list(pvals.keys())
    if not keys: return {}
    ps = np.array([pvals[k] for k in keys], dtype=float)
    rej, p_adj, _, _ = multipletests(ps, alpha=alpha, method="fdr_bh")
    return {k:(bool(rej[i]), float(p_adj[i])) for i,k in enumerate(keys)}

# =========================
# 7) 실행 루틴
# =========================
def _isi_grade(x):
    if pd.isna(x): return np.nan
    if x <= 7: return "정상"
    if x <= 14: return "경증"
    if x <= 21: return "중등도"
    return "중증"

def run_pipeline(pre_path: str,
                 post_path: str,
                 outdir: Path = Path("exp_outputs"),
                 id_col: str = ID_COL_DEFAULT,
                 debug: bool=False):
    outdir.mkdir(exist_ok=True, parents=True)

    # 로드 & 라벨
    pre, post = load_and_label(pre_path, post_path, id_col)

    # 디버그: ISI 응답 고유값
    if debug:
        print("\n[디버깅] ISI 문항 응답값 고유 목록:")
        isi_cols = find_cols(pre, PRE_ISI)
        for col in isi_cols:
            print(f"  - {col}: {pre[col].unique()}")

    # 스코어링
    pre_s, pre_info   = score_pre(pre, id_col)
    post_s, post_info = score_post(post, id_col)

    # 병합
    merged = pd.merge(pre_s, post_s, on=[id_col, "group"], how="inner")

    # ISI 총점/등급
    if "ISI_sev_pre" in merged.columns:
        merged["ISI_total_pre"] = merged["ISI_sev_pre"] * 7
        merged["ISI_grade_pre"] = merged["ISI_total_pre"].apply(_isi_grade)

    # Δ 계산
    blocks = [
        ("HBM_SEV_pre","HBM_SEV_post","HBM_SEV"),
        ("HBM_SUS_pre","HBM_SUS_post","HBM_SUS"),
        ("HBM_AWR_pre","HBM_AWR_post","HBM_AWR"),
        ("TPB_BB_pre","TPB_BB_post","TPB_BB"),
        ("TPB_SN_pre","TPB_SN_post","TPB_SN"),
        ("TPB_SE_pre","TPB_SE_post","TPB_SE"),
        ("TPB_INT_pre","TPB_INT_post","TPB_INT"),
        ("Stage_pre","Stage_post","Stage"),
    ]
    for pre_k, post_k, tag in blocks:
        if (pre_k in merged.columns) and (post_k in merged.columns):
            merged[f"{tag}_delta"] = merged[post_k] - merged[pre_k]

    # 저장물
    reliability = {**pre_info, **post_info}
    (outdir/"reliability.json").write_text(json.dumps(reliability, ensure_ascii=False, indent=2), encoding="utf-8")
    merged.to_csv(outdir/"merged_scored.csv", index=False, encoding="utf-8-sig")

    # 리포트
    report = []
    report.append("=== Controlled Experiment 1차 분석 리포트 (HBM 3분할 + PAQ-S 6문항·Jupyter) ===\n")
    report.append(f"N(사전∩사후)={len(merged)} | exp={int((merged['group']=='exp').sum())}, ctrl={int((merged['group']=='ctrl').sum())}\n")

    if "ISI_total_pre" in merged.columns:
        s = merged["ISI_total_pre"].dropna()
        report.append(f"\n[ISI(사전) 총점]\n- ISI_total_pre: M={s.mean():.2f}, SD={s.std(ddof=1):.2f}, N={len(s)}\n")
        report.append(f"- 등급 분포: {merged['ISI_grade_pre'].value_counts(dropna=False).to_dict()}\n")

    report.append("\n[기술통계]\n")
    tech_cols = [
        "ISI_sev_pre","ISI_sat_pre",
        "HBM_SEV_pre","HBM_SEV_post",
        "HBM_SUS_pre","HBM_SUS_post",
        "HBM_AWR_pre","HBM_AWR_post",
        "TPB_BB_pre","TPB_BB_post","TPB_SN_pre","TPB_SN_post",
        "TPB_SE_pre","TPB_SE_post","TPB_INT_pre","TPB_INT_post",
        "Stage_pre","Stage_post",
        "UEQS_post","UEQS_P_post","UEQS_H_post","SDT_post",
        "PAQ_total_pre","PAQ_DIF_neg","PAQ_DDF_neg","PAQ_EOT_neg",
        "PAQ_DIF_pos","PAQ_DDF_pos","PAQ_EOT_pos","PAQ_pre"
    ]
    for col in tech_cols:
        if col in merged.columns:
            s = merged[col].dropna()
            report.append(f"- {col}: M={s.mean():.2f}, SD={s.std(ddof=1):.2f}, N={len(s)}\n")

    # 혼합모형
    report.append(mixed_model_block(
        merged, id_col,
        dv_list=["HBM_SEV","HBM_SUS","HBM_AWR","TPB_INT","TPB_BB","TPB_SN","TPB_SE"],
        covariates=["ISI_sev_pre"]
    ))

    # Δ-검정 & ANCOVA
    report.append(ancova_delta_blocks(merged, blocks))

    # Stage 전이
    report.append(stage_transition_tests(merged, id_col, debug=debug))

    # UEQ-S/SDT
    report.append(ueq_s_analysis(merged))

    # --- NEW: PAQ ↔ Δ(HBM/TPB)
    report.append(paq_delta_association(merged))

    # FDR(BH) — 핵심 지표만
    pvals = {}
    for (pre, post, tag) in blocks:
        if all(c in merged.columns for c in [pre, post, "group"]):
            d = merged.dropna(subset=[pre, post, "group"]).copy()
            d["delta"] = d[post] - d[pre]
            if (d.groupby("group").size() >= 3).all():
                pvals[f"{tag}_delta_group"] = stats.ttest_ind(
                    d.loc[d.group=="exp","delta"], d.loc[d.group=="ctrl","delta"], equal_var=False
                )[1]
    for col in ["UEQS_post","UEQS_P_post","UEQS_H_post","SDT_post"]:
        if all(c in merged.columns for c in [col, "group"]):
            s = merged.dropna(subset=[col, "group"])
            if (s.groupby("group").size() >= 3).all():
                pvals[f"{col}_group"] = stats.ttest_ind(
                    s.loc[s.group=="exp", col], s.loc[s.group=="ctrl", col], equal_var=False
                )[1]
    if pvals:
        adj = multiple_testing_fdr(pvals, alpha=0.05)
        report.append("\n[FDR 보정(BH) - 핵심 지표]\n")
        for k,(rej,q) in adj.items():
            report.append(f"- {k}: q={q:.4f}, 유의={rej}\n")

    (outdir/"stats_report.txt").write_text("".join(report), encoding="utf-8")
    print("완료!")
    print(f"- 병합 데이터: {outdir/'merged_scored.csv'}")
    print(f"- 신뢰도: {outdir/'reliability.json'}")
    print(f"- 리포트: {outdir/'stats_report.txt'}")

# =========================
# 사용 예시 (경로만 맞춰 실행)
# =========================
run_pipeline(
    pre_path="data/cbt-i 사전 설문조사(응답).xlsx",
    post_path="data/cbt-i 사후 설문조사(응답).xlsx",
    outdir=Path("exp_outputs"),
    id_col=ID_COL_DEFAULT,
    debug=True
)

# ==== Cell 2 (code) ====
import json
data = json.load(open("exp_outputs/reliability.json", encoding="utf-8"))
print(json.dumps(data, indent=2, ensure_ascii=False))

# ==== Cell 3 (code) ====
print(open("exp_outputs/stats_report.txt", encoding="utf-8").read())

# ==== Cell 4 (code) ====
# Figure 2: Key interaction results (HBM_SEV, TPB_INT, Stage)
# - Requires: exp_outputs/merged_scored.csv (from your pipeline)
# - Output: ./figures/<redacted_token>.png
#           ./figures/<redacted_token>.png
#           ../path/to/file

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "exp_outputs/merged_scored.csv"
OUT_DIR  = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

def _t_crit(n):
    # simple approx: use 1.96 if n>=30 else Student t with df=n-1
    # (matplotlib-only constraint; avoiding scipy)
    if n is None or n < 2:
        return 1.96
    # small table fallback
    t_table = {
        1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
        18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045
    }
    df = max(1, n-1)
    return t_table.get(df, 1.96)

def summarize_ci(df, value_col, by_cols=("group","time")):
    g = df.groupby(list(by_cols))[value_col]
    mean = g.mean()
    sd   = g.std(ddof=1)
    n    = g.count()
    se   = sd / np.sqrt(n)
    tval = n.apply(_t_crit)
    ci   = se * tval
    out = pd.DataFrame({
        "mean": mean,
        "sd": sd,
        "n": n,
        "se": se,
        "ci95": ci
    }).reset_index()
    return out

def to_long(df, id_col, pre_col, post_col, value_name):
    keep = [id_col, "group", pre_col, post_col]
    s = df.dropna(subset=["group", pre_col, post_col]).copy()
    long = s.melt(id_vars=[id_col, "group"],
                  value_vars=[pre_col, post_col],
                  var_name="time", value_name=value_name)
    long["time"] = long["time"].map({pre_col: "pre", post_col: "post"})
    long["group"] = long["group"].map({"exp":"Exploratory", "ctrl":"Directive"}).fillna(long["group"])
    return long

def plot_interaction(long_df, value_col, y_label, title, outfile):
    summ = summarize_ci(long_df, value_col, by_cols=("group","time"))
    # ensure order
    time_order = ["pre","post"]
    group_order = ["Exploratory","Directive"] if set(summ["group"])>=set(["Exploratory","Directive"]) else sorted(summ["group"].unique())

    plt.figure(figsize=(4.5, 3.6), dpi=300)  # single-figure (no subplots)
    for grp in group_order:
        sub = summ[summ["group"]==grp].set_index("time").reindex(time_order)
        x = np.arange(len(time_order))
        y = sub["mean"].values
        yerr = sub["ci95"].values
        # Line + error bars (no explicit colors)
        plt.errorbar(x, y, yerr=yerr, fmt="-o", capsize=3, label=grp)

    plt.xticks(np.arange(len(time_order)), ["Pre", "Post"])
    plt.ylabel(y_label)
    plt.xlabel("Time")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, outfile), bbox_inches="tight")
    plt.close()

def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    # Try to detect id column automatically (same default used in your pipeline)
    id_candidates = [
        "1. 귀하의 성함이 어떻게 되십니까?",
        "id", "ID", "participant_id", "name"
    ]
    id_col = None
    for c in id_candidates:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        raise ValueError("ID column not found. Please set id_col manually.")

    # (a) HBM_SEV: Perceived Severity
    hbm_pre, hbm_post = "HBM_SEV_pre", "HBM_SEV_post"
    if hbm_pre in df.columns and hbm_post in df.columns:
        long_hbm = to_long(df, id_col, hbm_pre, hbm_post, "HBM_SEV")
        plot_interaction(
            long_hbm, "HBM_SEV",
            y_label="HBM Perceived Severity",
            title="(a) HBM Perceived Severity — Group × Time",
            outfile="<redacted_token>.png"
        )
    else:
        print("[WARN] HBM_SEV columns not found; skip (a)")

    # (b) TPB_INT: Intention
    tpb_pre, tpb_post = "TPB_INT_pre", "TPB_INT_post"
    if tpb_pre in df.columns and tpb_post in df.columns:
        long_tpb = to_long(df, id_col, tpb_pre, tpb_post, "TPB_INT")
        plot_interaction(
            long_tpb, "TPB_INT",
            y_label="TPB Intention",
            title="(b) TPB Intention — Group × Time",
            outfile="<redacted_token>.png"
        )
    else:
        print("[WARN] TPB_INT columns not found; skip (b)")

    # (c) Stage of Change (ordinal → mean with CI for compactness)
    stg_pre, stg_post = "Stage_pre", "Stage_post"
    if stg_pre in df.columns and stg_post in df.columns:
        long_stg = to_long(df, id_col, stg_pre, stg_post, "Stage")
        plot_interaction(
            long_stg, "Stage",
            y_label="Stage of Change (mean)",
            title="(c) Stage of Change — Group × Time",
            outfile="fig2c_stage_interaction.png"
        )
    else:
        print("[WARN] Stage columns not found; skip (c)")

    print("Saved:")
    for f in ["<redacted_token>.png",
              "<redacted_token>.png",
              "fig2c_stage_interaction.png"]:
        p = os.path.join(OUT_DIR, f)
        if os.path.exists(p):
            print(" -", p)

if __name__ == "__main__":
    main()

# ==== Cell 5 (code) ====
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("exp_outputs/merged_scored.csv", encoding="utf-8-sig")

groups = ["exp", "ctrl"]
labels = ["Exploratory", "Directive"]
means = [df.loc[df.group==g,"SDT_post"].mean() for g in groups]
ses   = [df.loc[df.group==g,"SDT_post"].std(ddof=1)/len(df.loc[df.group==g])**0.5 for g in groups]

plt.figure(figsize=(3.5,3))
plt.bar(labels, means, yerr=[1.96*s for s in ses], capsize=4)
plt.ylabel("Autonomy (SDT)")
plt.title("(a) SDT-based Autonomy")
plt.tight_layout()
plt.savefig("figures/sac_figure4a.png", dpi=300)

# ==== Cell 6 (code) ====
import pandas as pd, numpy as np, matplotlib.pyplot as plt

df = pd.read_csv("exp_outputs/merged_scored.csv", encoding="utf-8-sig")

metrics = ["UEQS_P_post", "UEQS_H_post"]
metric_labels = ["Pragmatic", "Hedonic"]
groups = ["exp", "ctrl"]
group_labels = ["Exploratory", "Directive"]

bar_width = 0.35
x = np.arange(len(metrics))

plt.figure(figsize=(4.2, 3.2), dpi=300)

for i, g in enumerate(groups):
    vals, ses = [], []
    for m in metrics:
        d = df.loc[df.group == g, m].dropna()
        vals.append(d.mean())
        ses.append(d.std(ddof=1)/np.sqrt(len(d)))
    plt.bar(x + (i-0.5)*bar_width, vals, bar_width,
            yerr=[1.96*s for s in ses], capsize=3,
            label=group_labels[i])

plt.xticks(x, metric_labels)
plt.ylabel("UEQ-S score")
plt.ylim(0, 6)
plt.title("(b) UEQ-S Pragmatic vs Hedonic")
plt.legend(frameon=True, facecolor='white', edgecolor='lightgray', framealpha=1.0)
plt.tight_layout()
plt.savefig("figures/sac_figure4b.png", dpi=300)
plt.show()


# ==== Cell 7 (code) ====
# fig4_moderation_only.py
# Input: exp_outputs/merged_scored.csv
# Output: figures/sac_figure5a.png, figures/sac_figure5b.png

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

CSV = "exp_outputs/merged_scored.csv"
OUT = Path("figures"); OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV, encoding="utf-8-sig")

# ---- 준비: 필요한 열이 모두 있음 (네 파일 컬럼 목록 기준)
#   DIF: PAQ_DIF_neg, DDF: PAQ_DDF_neg, EOT: PAQ_EOT_neg
#   ΔSeverity: HBM_SEV_delta, ΔOutcome beliefs: TPB_BB_delta

def _linreg_line(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 3: return None
    b1, b0 = np.polyfit(x[m].astype(float), y[m].astype(float), 1)
    xs = np.linspace(x[m].min(), x[m].max(), 100)
    ys = b1*xs + b0
    return xs, ys

def _ci95(series):
    s = pd.Series(series).dropna()
    if len(s) < 2: return np.nan
    se = s.std(ddof=1)/np.sqrt(len(s))
    return 1.96*se

# (a) Scatter: DIF/DDF vs ΔSeverity (회귀선)
plt.figure(figsize=(4.2,3.4), dpi=300)
plotted = False

for col, lab in [("PAQ_DIF_neg","DIF"), ("PAQ_DDF_neg","DDF")]:
    if col in df.columns and "HBM_SEV_delta" in df.columns:
        x, y = df[col], df["HBM_SEV_delta"]
        plt.scatter(x, y, s=18, alpha=.85, label=lab)
        lr = _linreg_line(x,y)
        if lr: plt.plot(*lr)
        plotted = True

if plotted:
    plt.axhline(0, lw=1)
    plt.xlabel("Alexithymia (DIF/DDF)")
    plt.ylabel("Δ Perceived Severity (post − pre)")
    plt.title("(a) DIF/DDF vs. ΔSeverity")
    plt.legend(frameon=True, facecolor="white", edgecolor="lightgray", framealpha=1.0)
    plt.tight_layout()
    plt.savefig(OUT/"sac_figure5a.png", bbox_inches="tight")
plt.close()

# (b) Interaction: EOT High vs Low → ΔTPB_BB
if {"PAQ_EOT_neg","TPB_BB_delta","group"}.issubset(df.columns):
    thr = df["PAQ_EOT_neg"].median()
    df["_EOTgrp"] = np.where(df["PAQ_EOT_neg"] <= thr, "Low EOT", "High EOT")
    df["_Group"] = df["group"].map({"exp":"Exploratory","ctrl":"Directive"}).fillna(df["group"])

    g = df.groupby(["_EOTgrp","_Group"])["TPB_BB_delta"]
    mean = g.mean()
    ci = g.apply(_ci95)
    summary = pd.DataFrame({"mean":mean, "ci":ci}).reset_index()

    import numpy as np
    xlabels = ["Low EOT","High EOT"]; x = np.arange(2)
    plt.figure(figsize=(4.4,3.4), dpi=300)
    for i, grp in enumerate(["Exploratory","Directive"]):
        sub = summary[summary["_Group"]==grp].set_index("_EOTgrp").reindex(xlabels)
        plt.errorbar(x + (i-0.5)*0.05, sub["mean"].values, yerr=sub["ci"].values,
                     fmt="-o", capsize=3, label=grp)
    plt.xticks(x, xlabels)
    plt.ylabel("Outcome Belief Gain (Δ TPB_BB)")
    plt.title("(b) EOT (High vs Low) → Outcome Belief Gain")
    plt.legend(frameon=True, facecolor="white", edgecolor="lightgray", framealpha=1.0)
    plt.tight_layout()
    plt.savefig(OUT/"sac_figure5b.png", bbox_inches="tight")
    plt.close()

print("Figure 4 done → figures/sac_figure5a.png, figures/sac_figure5b.png")

# ==== Cell 8 (code) ====
# <redacted_token>.py
# Input : exp_outputs/merged_scored.csv
# Output: figures/sac_figure5a1.png, sac_figure5a2.png, sac_figure5b.png, sac_figure5_abc.png

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

CSV = "exp_outputs/merged_scored.csv"
OUT = Path("figures"); OUT.mkdir(parents=True, exist_ok=True)

Y_FLIP = False  # True로 바꾸면 "위로 갈수록 개선" (Improvement = -(post-pre))

df = pd.read_csv(CSV, encoding="utf-8-sig")

X_VARS = [("PAQ_DIF_neg", "DIF"), ("PAQ_DDF_neg", "DDF")]
Y_COL = "HBM_SEV_delta"

# ---- helpers
def _pearson_r(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 2: return np.nan, m.sum()
    r = np.corrcoef(x[m].astype(float), y[m].astype(float))[0,1]
    return r, m.sum()

def _regline(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 3: return None
    b1, b0 = np.polyfit(x[m].astype(float), y[m].astype(float), 1)
    xs = np.linspace(x[m].min(), x[m].max(), 100)
    ys = b1*xs + b0
    return xs, ys

def _ci95(s):
    s = pd.Series(s).dropna()
    if len(s) < 2: return np.nan
    se = s.std(ddof=1) / np.sqrt(len(s))
    return 1.96 * se

def _bin_summary(x, y, bins=3):
    m = x.notna() & y.notna()
    xv, yv = x[m].astype(float).values, y[m].astype(float).values
    if len(xv) < 3: return None
    qs = np.quantile(xv, np.linspace(0,1,bins+1))
    qs[0] -= 1e-6; qs[-1] += 1e-6
    centers, means, ci95 = [], [], []
    for i in range(bins):
        mask = (xv>=qs[i]) & (xv<qs[i+1])
        if mask.sum() >= 2:
            xb = xv[mask]; yb = yv[mask]
            centers.append(xb.mean())
            m = yb.mean(); se = yb.std(ddof=1)/np.sqrt(len(yb))
            means.append(m); ci95.append(1.96*se)
    if not centers: return None
    return np.array(centers), np.array(means), np.array(ci95)

def plot_panel(ax, x, y, xlabel, title_prefix):
    # 산점도
    ax.scatter(x, y, s=20, alpha=0.9)
    # 회귀선
    rl = _regline(x, y)
    if rl: ax.plot(*rl, linewidth=2)
    # 3구간 평균 ±95%CI
    bs = _bin_summary(x, y, bins=3)
    if bs:
        cx, my, ci = bs
        ax.errorbar(cx, my, yerr=ci, fmt="s", capsize=3, linewidth=1.5)
    # 기준선
    if not Y_FLIP:
        ax.axhline(0, color="k", linewidth=1)
    # 주석
    r, n = _pearson_r(x, y)
    ax.text(0.05, 0.95, f"r = {r:.2f}, N = {n}", transform=ax.transAxes,
            ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="lightgray", boxstyle="round,pad=0.2"))
    # 라벨
    ax.set_xlabel(f"Alexithymia ({xlabel})")
    ax.set_title(f"{title_prefix} {xlabel} vs. " + ("Improvement" if Y_FLIP else "ΔSeverity"))

# Y축 방향 설정
if Y_FLIP:
    df["_Y"] = -(df[Y_COL])
    y_label = "Improvement in Severity (−Δ = pre − post)"  # ✅ 짧게 수정
else:
    df["_Y"] = df[Y_COL]
    y_label = "Δ Severity (post − pre)"  # ✅ 짧게 수정

# 공통 y-축 범위(두 패널 동일)
y_pool = df["_Y"].dropna()
if len(y_pool):
    pad = 0.1 * (y_pool.max() - y_pool.min() + 1e-6)
    YLIM = (y_pool.min() - pad, y_pool.max() + pad)
else:
    YLIM = None

# ---------- (a) DIF 개별 파일
x = df[X_VARS[0][0]]; y = df["_Y"]
fig, ax = plt.subplots(figsize=(4.4, 3.4), dpi=300)
plot_panel(ax, x, y, X_VARS[0][1], "(a)")
ax.set_ylabel(y_label)
if YLIM: ax.set_ylim(*YLIM)
plt.tight_layout(); plt.savefig(OUT/"sac_figure5a1.png", bbox_inches="tight"); plt.close(fig)

# ---------- (b) DDF 개별 파일
x = df[X_VARS[1][0]]; y = df["_Y"]
fig, ax = plt.subplots(figsize=(4.4, 3.4), dpi=300)
plot_panel(ax, x, y, X_VARS[1][1], "(b)")
ax.set_ylabel(y_label)
if YLIM: ax.set_ylim(*YLIM)
plt.tight_layout(); plt.savefig(OUT/"sac_figure5a2.png", bbox_inches="tight"); plt.close(fig)

# ---------- (c) EOT High vs Low → Outcome Belief Gain  (재생성)
if {"PAQ_EOT_neg","TPB_BB_delta","group"}.issubset(df.columns):
    thr = df["PAQ_EOT_neg"].median()
    df["_EOTgrp"] = np.where(df["PAQ_EOT_neg"] <= thr, "Low EOT", "High EOT")
    df["_Group"]  = df["group"].map({"exp":"Exploratory","ctrl":"Directive"}).fillna(df["group"])
    g = df.groupby(["_EOTgrp","_Group"])["TPB_BB_delta"]
    mean = g.mean(); ci = g.apply(_ci95)
    summary = pd.DataFrame({"mean":mean, "ci":ci}).reset_index()

    xlabels = ["Low EOT","High EOT"]; xpos = np.arange(2)
    fig, ax = plt.subplots(figsize=(4.4, 3.4), dpi=300)
    for i, grp in enumerate(["Exploratory","Directive"]):
        sub = summary[summary["_Group"]==grp].set_index("_EOTgrp").reindex(xlabels)
        ax.errorbar(xpos + (i-0.5)*0.05, sub["mean"].values, yerr=sub["ci"].values,
                    fmt="-o", capsize=3, label=grp)
    ax.set_xticks(xpos); ax.set_xticklabels(xlabels)
    ax.set_ylabel("Outcome Belief Gain (Δ TPB_BB)")
    ax.set_title("(c) EOT (High vs Low) → Outcome Belief Gain")
    ax.legend(frameon=True, facecolor="white", edgecolor="lightgray", framealpha=1.0)
    plt.tight_layout(); plt.savefig(OUT/"sac_figure5b.png", bbox_inches="tight"); plt.close(fig)

# ---------- 3패널 합본 (a+b+c)
fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.6), dpi=300)
# a
x = df[X_VARS[0][0]]; y = df["_Y"]
plot_panel(axes[0], x, y, X_VARS[0][1], "(a)"); axes[0].set_ylabel(y_label)
if YLIM: axes[0].set_ylim(*YLIM)
# b
x = df[X_VARS[1][0]]; y = df["_Y"]
plot_panel(axes[1], x, y, X_VARS[1][1], "(b)")
if YLIM: axes[1].set_ylim(*YLIM)
# c (EOT)
if {"PAQ_EOT_neg","TPB_BB_delta","group"}.issubset(df.columns):
    thr = df["PAQ_EOT_neg"].median()
    df["_EOTgrp"] = np.where(df["PAQ_EOT_neg"] <= thr, "Low EOT", "High EOT")
    df["_Group"]  = df["group"].map({"exp":"Exploratory","ctrl":"Directive"}).fillna(df["group"])
    g = df.groupby(["_EOTgrp","_Group"])["TPB_BB_delta"]
    mean = g.mean(); ci = g.apply(_ci95)
    summary = pd.DataFrame({"mean":mean, "ci":ci}).reset_index()
    xlabels = ["Low EOT","High EOT"]; xpos = np.arange(2)
    axes[2].errorbar(xpos-0.025, summary[summary["_Group"]=="Exploratory"].set_index("_EOTgrp").reindex(xlabels)["mean"].values,
                     yerr=summary[summary["_Group"]=="Exploratory"].set_index("_EOTgrp").reindex(xlabels)["ci"].values,
                     fmt="-o", capsize=3, label="Exploratory")
    axes[2].errorbar(xpos+0.025, summary[summary["_Group"]=="Directive"].set_index("_EOTgrp").reindex(xlabels)["mean"].values,
                     yerr=summary[summary["_Group"]=="Directive"].set_index("_EOTgrp").reindex(xlabels)["ci"].values,
                     fmt="-o", capsize=3, label="Directive")
    axes[2].set_xticks(xpos); axes[2].set_xticklabels(xlabels)
    axes[2].set_title("(c) EOT (High vs Low) → Outcome Belief Gain")
    axes[2].legend(frameon=True, facecolor="white", edgecolor="lightgray", framealpha=1.0)
plt.tight_layout()
plt.savefig(OUT/"sac_figure5_abc.png", bbox_inches="tight")
plt.close(fig)

print("Save.\path\to\file
