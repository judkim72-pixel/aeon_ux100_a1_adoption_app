
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="UX100 · A-2 · Model/Stack Disclosure & Responsible AI", layout="wide")
st.title("UX100 · A-2 · 모델/스택 공개 & Responsible AI")
st.markdown("<span style='font-size:12px;color:#6b7280'>UX100 · A-2 · Model/Stack Disclosure & Responsible AI</span>", unsafe_allow_html=True)

# ---------- Load data (auto from repo root if available) ----------
DEFAULT_DATA_PATH = "ux_100_dataset.xlsx"
df = None
auto_loaded = False

if os.path.exists(DEFAULT_DATA_PATH):
    try:
        df = pd.read_excel(DEFAULT_DATA_PATH, sheet_name="Data")
        auto_loaded = True
        st.success("기본 데이터셋을 불러왔습니다: ux_100_dataset.xlsx")
        st.markdown("<span style='font-size:12px;color:#6b7280'>Loaded default dataset: ux_100_dataset.xlsx</span>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"루트에 파일이 있으나 읽기에 실패했습니다: {e}")
        st.markdown("<span style='font-size:12px;color:#6b7280'>Found dataset at root but failed to read.</span>", unsafe_allow_html=True)

if df is None:
    st.markdown("엑셀 파일(.xlsx)을 업로드하거나, 서버 경로를 입력해 주세요. (시트명은 **Data**)")
    st.markdown("<span style='font-size:12px;color:#6b7280'>Upload Excel (.xlsx) or type a server path. Sheet must be named <b>Data</b>.</span>", unsafe_allow_html=True)
    uploaded = st.file_uploader("엑셀 업로드 (.xlsx)", type=["xlsx"])
    path_in = st.text_input("또는 서버 경로 입력", value="")

    @st.cache_data
    def load_df_from_upload(file):
        return pd.read_excel(file, sheet_name="Data")

    @st.cache_data
    def load_df_from_path(path):
        return pd.read_excel(path, sheet_name="Data")

    if uploaded is not None:
        try:
            df = load_df_from_upload(uploaded)
        except Exception as e:
            st.error(f"업로드 파일 읽기 실패: {e}")
    elif path_in:
        try:
            df = load_df_from_path(path_in)
        except Exception as e:
            st.error(f"경로 읽기 실패: {e}")

if df is None:
    st.info("데이터 대기 중… (시트명은 'Data', 필수 컬럼: 'Company')")
    st.stop()

# ---------- Helpers ----------
def yes_like(series):
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["y","yes","true","1","public","open","available"])

def non_empty(series):
    return series.astype(str).str.strip().ne("")

def contains_any(text, keywords):
    t = str(text).lower()
    return any(k in t for k in keywords)

# ---------- Signals ----------
col_disclosure = None
for c in df.columns:
    if "model" in str(c).lower() and "disclosure" in str(c).lower():
        col_disclosure = c; break
if col_disclosure is None:
    # fallback: try 'Model/Stack Disclosure' exactly
    if "Model/Stack Disclosure" in df.columns:
        col_disclosure = "Model/Stack Disclosure"

col_ethics = None
for c in df.columns:
    if "ethics" in str(c).lower() or ("privacy" in str(c).lower() and "ai" in str(c).lower()):
        col_ethics = c; break
if col_ethics is None and "Privacy/AI Ethics Policy (public Y/N)" in df.columns:
    col_ethics = "Privacy/AI Ethics Policy (public Y/N)"

# Booleans
disclosed = pd.Series(False, index=df.index)
if col_disclosure is not None:
    disclosed = non_empty(df[col_disclosure]) & (~df[col_disclosure].astype(str).str.contains("not disclosed|n/a|na|none", case=False, na=False))

ethics = pd.Series(False, index=df.index)
if col_ethics is not None:
    ethics = yes_like(df[col_ethics]) | (non_empty(df[col_ethics]) & (~df[col_ethics].astype(str).str.contains("no|not|none", case=False, na=False)))

# Depth heuristic for disclosure
depth_keywords_lvl3 = ["gpt","gpt-","llm","llama","mixtral","mistral","claude","gemini","bert","t5","vicuna","falcon","qwen","rwkv","whisper","sdxl","lora","sft","rlhf","rlaif","pretrain","fine-tune","rag","vector","embedding","semantic","rerank","retrieval","sagemaker","vertex","azure openai","openai","huggingface","model card","eval","hallucination","guardrail"]
depth = []
for i, row in df.iterrows():
    txt = str(row[col_disclosure]) if col_disclosure else ""
    if not txt.strip():
        depth.append(0)
    else:
        # level 1: says disclosed but no details
        lvl = 1
        if contains_any(txt, ["provider","cloud","on-prem","self-host","api","sdk","pipeline","mle","mlops","governance"]):
            lvl = max(lvl, 2)
        if contains_any(txt, depth_keywords_lvl3):
            lvl = max(lvl, 3)
        depth.append(lvl)
depth = pd.Series(depth, index=df.index)

# A-2 composite index (0–100)
# weights: disclosure 50, depth 30, ethics 20
a2 = (disclosed.astype(float)*50.0) + (depth.astype(float)/3.0*30.0) + (ethics.astype(float)*20.0)
a2 = a2.round(1)

work = pd.DataFrame({
    "Company": df["Company"],
    "Disclosure_Flag": disclosed,
    "Disclosure_Depth(0-3)": depth,
    "Ethics_Flag": ethics,
    "A2_Index(0-100)": a2
})

# ---------- Chart 1: A-2 Index distribution ----------
st.subheader("분포(Distribution) — A-2 지수 (0–100)")
st.markdown("<span style='font-size:12px;color:#6b7280'>Histogram of A-2 Index: Model/Stack disclosure × depth × Responsible AI policy.</span>", unsafe_allow_html=True)
bins = st.slider("구간 수 (bins)", 5, 30, 12, 1)
fig = px.histogram(work, x="A2_Index(0-100)", nbins=bins, opacity=0.9)
p25 = float(work["A2_Index(0-100)"].quantile(0.25))
p50 = float(work["A2_Index(0-100)"].quantile(0.50))
p75 = float(work["A2_Index(0-100)"].quantile(0.75))
fig.add_vrect(x0=p25, x1=p75, fillcolor="LightSkyBlue", opacity=0.15, line_width=0)
fig.add_vline(x=p25, line_dash="dash", line_color="SteelBlue")
fig.add_vline(x=p50, line_dash="dot", line_color="SlateGray")
fig.add_vline(x=p75, line_dash="dash", line_color="SteelBlue")
fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=420, xaxis_title="A-2 Index (0–100)")
st.plotly_chart(fig, use_container_width=True)
st.markdown("- **해설:** A-2는 공개(50)·공개 깊이(30)·윤리 정책(20)의 가중합입니다. IQR(음영)과 중앙값(점선)으로 분포의 중심과 꼬리를 함께 파악합니다.")
st.markdown("<span style='font-size:12px;color:#6b7280'>Explanation: A-2 is a weighted sum of disclosure (50), depth (30), and ethics policy (20). IQR shading and median line help interpret the center and tails.</span>", unsafe_allow_html=True)

# ECDF (optional)
if st.toggle("누적분포(ECDF) 보기 / Show ECDF", key="ecdf_a2"):
    ecdf = px.ecdf(work, x="A2_Index(0-100)")
    ecdf.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=260, xaxis_title="A-2 Index (0–100)")
    st.plotly_chart(ecdf, use_container_width=True)
    st.markdown("- **해설:** ECDF 급경사 구간은 동일/유사 설정을 공유하는 기업이 몰려 있음을 의미합니다.")
    st.markdown("<span style='font-size:12px;color:#6b7280'>ECDF steep sections indicate crowded configurations.</span>", unsafe_allow_html=True)

# ---------- Chart 2: 2×2 heatmap (Disclosure × Ethics) ----------
st.subheader("2×2 매트릭스 — 공개 × 윤리 정책")
st.markdown("<span style='font-size:12px;color:#6b7280'>Quadrant counts: Disclosure (Yes/No) × Ethics policy (Yes/No).</span>", unsafe_allow_html=True)

quad = pd.crosstab(work["Disclosure_Flag"].map({True:"공개 Yes", False:"공개 No"}),
                   work["Ethics_Flag"].map({True:"윤리 Yes", False:"윤리 No"}))

heat = px.imshow(quad.values, labels=dict(x="윤리 정책", y="모델/스택 공개", color="기업 수"),
                 x=list(quad.columns), y=list(quad.index), text_auto=True, aspect="auto")
heat.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=360)
st.plotly_chart(heat, use_container_width=True)
st.markdown("- **해설:** 우상단(공개 Yes & 윤리 Yes)은 엔터프라이즈 **신뢰성 신호**가 높은 구간입니다. 좌하단은 개선여지가 큽니다.")
st.markdown("<span style='font-size:12px;color:#6b7280'>Explanation: Top-right quadrant combines high disclosure and ethics — a strong enterprise signal. Bottom-left indicates room for improvement.</span>", unsafe_allow_html=True)

# ---------- Chart 3: Company-level scatter (A-2) ----------
st.subheader("기업별 분포(Company-level) — A-2 지수")
st.markdown("<span style='font-size:12px;color:#6b7280'>Scatter with jitter; hover shows company and A-2, plus flags.</span>", unsafe_allow_html=True)
np.random.seed(42)
jitter = np.random.uniform(-0.05, 0.05, size=len(work))
work["_y"] = jitter
sc = px.scatter(work.sort_values("A2_Index(0-100)"),
                x="A2_Index(0-100)", y="_y",
                hover_name="Company",
                hover_data={"A2_Index(0-100)":":.1f",
                            "Disclosure_Flag":True,
                            "Disclosure_Depth(0-3)":True,
                            "Ethics_Flag":True,
                            "_y":False})
sc.update_layout(yaxis_title="jitter (no metric)",
                 yaxis={"visible": True, "showticklabels": False},
                 margin=dict(l=10,r=10,t=10,b=10), height=340,
                 xaxis_title="A-2 Index (0–100)")
st.plotly_chart(sc, use_container_width=True)
st.markdown("- **해설:** 세로축은 값의 의미가 없고(지터), 가로축 A-2 점수만 의미합니다.")
st.markdown("<span style='font-size:12px;color:#6b7280'>Explanation: Vertical axis has no metric; A-2 score on the horizontal axis carries meaning.</span>", unsafe_allow_html=True)

# ---------- Table: Top / Bottom ----------
st.subheader("상·하위 기업 표 — A-2 지수 기준")
top_n = st.number_input("상·하위 표시 개수", min_value=3, max_value=20, value=10, step=1, key="tb_a2")
top = work.sort_values("A2_Index(0-100)", ascending=False).head(top_n)
bot = work.sort_values("A2_Index(0-100)", ascending=True).head(top_n)
st.write("**상위 기업(Top companies)**"); st.dataframe(top.reset_index(drop=True))
st.write("**하위 기업(Bottom companies)**"); st.dataframe(bot.reset_index(drop=True))
st.markdown("- **해설:** 상위권은 상세 공개와 윤리 정책을 **동시에** 충족하는 경향. 하위권은 둘 다 부재하거나 깊이가 낮습니다.")
st.markdown("<span style='font-size:12px;color:#6b7280'>Explanation: Top cohort tends to meet both detailed disclosure and ethics; bottom cohort lacks one or both.</span>", unsafe_allow_html=True)

# ---------- Notes ----------
if auto_loaded:
    st.caption("루트의 데이터셋(ux_100_dataset.xlsx)에서 자동 로드되었습니다.")
else:
    st.caption("스냅샷 기반 분석입니다. 월/분기 단위로 동일 필드를 수집하면 추세 분석이 가능합니다.")
