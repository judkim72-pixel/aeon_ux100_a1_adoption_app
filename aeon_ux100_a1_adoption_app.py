
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="UX100 · A-1 · AI Adoption Score Distribution", layout="wide")
st.title("UX100 · A-1 · AI Adoption Score Distribution")
st.markdown("<span style='font-size:13px;color:#6b7280'>UX100 · A-1 · AI Adoption Score Distribution</span>", unsafe_allow_html=True)

# --- Auto-load root dataset if present ---
DEFAULT_DATA_PATH = "ux_100_dataset.xlsx"  # repo root filename
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

# --- Fallback UI only when auto-load not available ---
if df is None:
    st.markdown("엑셀 파일(.xlsx)을 업로드하거나, 서버 경로를 입력해 주세요. (시트명은 **Data**)")
    st.markdown("<span style='font-size:12px;color:#6b7280'>Upload Excel (.xlsx) or type a server path. Sheet must be named <b>Data</b>.</span>", unsafe_allow_html=True)
    uploaded = st.file_uploader("엑셀 업로드 (.xlsx)", type=["xlsx"])
    path_in = st.text_input("또는 서버 경로 입력", value="", help="업로드를 사용할 경우 비워 두세요.")

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
            st.markdown("<span style='font-size:12px;color:#6b7280'>Failed to read uploaded file.</span>", unsafe_allow_html=True)
    elif path_in:
        try:
            df = load_df_from_path(path_in)
        except Exception as e:
            st.error(f"경로 읽기 실패: {e}")
            st.markdown("<span style='font-size:12px;color:#6b7280'>Failed to read file from path.</span>", unsafe_allow_html=True)

if df is None:
    st.info("데이터 대기 중… (시트명은 'Data', 필수 컬럼: 'Company', 'AI Adoption Index (0–5)')")
    st.markdown("<span style='font-size:12px;color:#6b7280'>Waiting for data… Required columns: Company, AI Adoption Index (0–5).</span>", unsafe_allow_html=True)
    st.stop()

# --- Identify core columns ---
ai_candidates = [c for c in df.columns if "AI Adoption Index" in str(c)]
if not ai_candidates:
    st.error("'Data' 시트에서 'AI Adoption Index (0–5)' 컬럼을 찾지 못했습니다.")
    st.markdown("<span style='font-size:12px;color:#6b7280'>Column 'AI Adoption Index (0–5)' not found.</span>", unsafe_allow_html=True)
    st.stop()
ai_col = ai_candidates[0]

df[ai_col] = pd.to_numeric(df[ai_col], errors="coerce")

# Optional bonus signals
def has(col, positive_values=None):
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[col].astype(str).str.strip().str.lower()
    if positive_values is None:
        return s.ne("")
    pv = [str(v).lower() for v in positive_values]
    return s.isin(pv)

model_bonus = has("Model/Stack Disclosure") & (~df["Model/Stack Disclosure"].astype(str).str.contains("not disclosed", case=False, na=False))
oss_bonus   = has("OSS Contributions", ["contributor", "maintainer"])
blog_bonus  = has("Blog Cadence") & (df["Blog Cadence"].astype(str).str.lower().ne("none"))
ethics_bonus = has("Privacy/AI Ethics Policy (public Y/N)", ["y", "yes", "true", "1"])
analytics_bonus = has("Analytics/Experimentation")

# --- Build 1–10 granular score ---
base = df[ai_col].fillna(0).astype(float)                 # 0–5 scale
score10 = 1 + 1.6*base                                    # 0–8 mapped + 1 floor
score10 += 0.6*model_bonus.astype(float)
score10 += 0.4*oss_bonus.astype(float)
score10 += 0.4*blog_bonus.astype(float)
score10 += 0.2*ethics_bonus.astype(float)
score10 += 0.4*analytics_bonus.astype(float)

score10 = score10.clip(1, 10).round(1)                    # clamp & 0.1 precision
df["_AI_Adoption_10"] = score10

dfv = df[["Company", ai_col, "_AI_Adoption_10"]].dropna().copy()
dfv.rename(columns={ai_col: "AI_Adoption_5"}, inplace=True)

# --- Distribution: Histogram with hover (1–10 score) ---
st.subheader("분포(Distribution) — 1–10 세분화 점수")
st.markdown("<span style='font-size:12px;color:#6b7280'>Histogram of derived 1–10 AI adoption scores. Hover to see company & values.</span>", unsafe_allow_html=True)
bins = st.slider("구간 수 (bins)", min_value=5, max_value=30, value=12, step=1)
fig = px.histogram(
    dfv,
    x="_AI_Adoption_10",
    nbins=bins,
    hover_data={"Company": True, "_AI_Adoption_10": ":.1f", "AI_Adoption_5": ":.1f"},
    opacity=0.85
)
fig.update_traces(marker_line_width=0.5)
fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=420, xaxis_title="AI Adoption (1–10)")
st.plotly_chart(fig, use_container_width=True)

# --- Company-level view: scatter with jitter (fix for blank strip plot) ---
st.subheader("기업별 분포 시각화(Company-level view) — 1–10 점수")
st.markdown("<span style='font-size:12px;color:#6b7280'>Scatter with slight jitter; hover for company & values.</span>", unsafe_allow_html=True)

np.random.seed(7)
jitter = np.random.uniform(-0.05, 0.05, size=len(dfv))
dfv["_y"] = 0.0 + jitter

fig2 = px.scatter(
    dfv.sort_values("_AI_Adoption_10"),
    x="_AI_Adoption_10",
    y="_y",
    hover_name="Company",
    hover_data={"_AI_Adoption_10":":.1f", "AI_Adoption_5":":.1f", "_y": False},
)
fig2.update_layout(
    yaxis={"visible": False},
    margin=dict(l=10,r=10,t=10,b=10),
    height=320,
    xaxis_title="AI Adoption (1–10)"
)
st.plotly_chart(fig2, use_container_width=True)

# --- Summary stats (new score) ---
st.subheader("요약 통계(Summary stats) — 1–10 점수")
desc = dfv["_AI_Adoption_10"].describe()
left, right = st.columns(2)
with left:
    st.metric("유효 표본 수", int(desc["count"]))
    st.metric("평균", f"{desc['mean']:.2f}")
    st.metric("중앙값", f"{desc['50%']:.2f}")
with right:
    st.metric("표준편차", f"{desc['std']:.2f}")
    st.metric("최솟값", f"{desc['min']:.1f}")
    st.metric("최댓값", f"{desc['max']:.1f}")
st.markdown("<span style='font-size:12px;color:#6b7280'>N (valid), Mean, Median, Std, Min, Max computed on the 1–10 derived score.</span>", unsafe_allow_html=True)

# --- Top & Bottom cohorts (by 1–10 score) ---
st.subheader("상위·하위 그룹(Top & Bottom cohorts) — 1–10 점수 기준")
top_n = st.number_input("상·하위 표시 개수", min_value=3, max_value=15, value=10, step=1)
top = dfv.sort_values("_AI_Adoption_10", ascending=False).head(top_n)[["Company","_AI_Adoption_10","AI_Adoption_5"]]
bot = dfv.sort_values("_AI_Adoption_10", ascending=True).head(top_n)[["Company","_AI_Adoption_10","AI_Adoption_5"]]

st.write("**상위 기업(Top companies)**")
st.dataframe(top.reset_index(drop=True))
st.write("**하위 기업(Bottom companies)**")
st.dataframe(bot.reset_index(drop=True))

# --- Narrative below charts ---
st.markdown("### 데이터가 보여주는 바 (What the data shows)")
p75 = dfv["_AI_Adoption_10"].quantile(0.75); p25 = dfv["_AI_Adoption_10"].quantile(0.25)
st.markdown(
    f"- 1–10 점수의 중심 경향은 **평균 {desc['mean']:.2f} / 중앙값 {desc['50%']:.2f}**이며, 사분위 범위(IQR)는 **{p25:.2f}–{p75:.2f}**입니다.\n"
    f"- **상위 {top_n}** 기업은 공개성/오픈 활동/거버넌스 신호에 따른 가점으로 우측 꼬리에 분포하는 경향이 있습니다."
)
st.markdown("<span style='font-size:12px;color:#6b7280'>Central tendency and IQR on the 1–10 derived score; right-tail clustering reflects disclosure/open/governance bonuses.</span>", unsafe_allow_html=True)

# --- Keywords ---
st.markdown("### 키워드 (Keywords)")
st.write(", ".join([
    "AI 도입(1–10)", "모델/스택 공개 가점", "OSS/블로그 가점", "윤리/분석 가점",
    "엔터프라이즈 적합성", "거버넌스", "디자인옵스"
]))
st.markdown("<span style='font-size:12px;color:#6b7280'>AI adoption (1–10), model/stack disclosure, OSS/blog, ethics/analytics, enterprise readiness, governance, design ops.</span>", unsafe_allow_html=True)

# --- Insights ---
st.markdown("### 인사이트 (Insights)")
st.markdown(
    "1) **세분화 점수로 상·중·하 변별력 확대.** 공개·오픈·거버넌스 신호가 실제 채택 성숙도의 차이를 잘 설명합니다.\n"
    "2) **보너스 신호의 축적 효과.** 모델 공개 + OSS + 블로그 + 윤리/분석이 겹칠수록 1–10 점수는 유의미하게 상승합니다.\n"
    "3) **하위권 개선 포인트 명확.** 단순 도입(5점 스케일)에서 멈춘 조직은 *공개/거버넌스/오픈 활동* 강화로 곧바로 점수 개선 여지가 있습니다."
)
st.markdown("<span style='font-size:12px;color:#6b7280'>1) Granularity improves separation. 2) Cumulative bonus effects. 3) Clear levers for lower cohort improvement.</span>", unsafe_allow_html=True)

# --- Top vs Bottom bullets ---
st.markdown("### 상위 vs 하위 특징 (Top vs. Bottom differences)")
def bulletize(dfpart, label_kr, label_en):
    names = " · ".join(dfpart["Company"].head(7).tolist())
    st.markdown(f"- **{label_kr}**: {names}" + (" …" if len(dfpart)>7 else ""))
    st.markdown(f"<span style='font-size:12px;color:#6b7280'>- <b>{label_en}</b>: {names}" + (" …" if len(dfpart)>7 else "") + "</span>", unsafe_allow_html=True)

bulletize(top, "상위 그룹", "Top cohort")
bulletize(bot, "하위 그룹", "Bottom cohort")

if auto_loaded:
    st.caption("루트의 데이터셋(ux_100_dataset.xlsx)에서 자동 로드되었습니다. 자동 로드 시 업로드 UI는 숨겨집니다.")
    st.markdown("<span style='font-size:12px;color:#6b7280'>Loaded from repo-root dataset; upload UI hidden on auto-load.</span>", unsafe_allow_html=True)
else:
    st.caption("스냅샷 기반 분석입니다. 시계열 추세는 월/분기별 동일 필드 캡처로 확장하세요.")
    st.markdown("<span style='font-size:12px;color:#6b7280'>Snapshot-only. For trends, capture the same fields monthly/quarterly.</span>", unsafe_allow_html=True)
