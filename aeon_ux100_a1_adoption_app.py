
import os
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

# --- Identify AI column ---
ai_candidates = [c for c in df.columns if "AI Adoption Index" in str(c)]
if not ai_candidates:
    st.error("'Data' 시트에서 'AI Adoption Index (0–5)' 컬럼을 찾지 못했습니다.")
    st.markdown("<span style='font-size:12px;color:#6b7280'>Column 'AI Adoption Index (0–5)' not found.</span>", unsafe_allow_html=True)
    st.stop()
ai_col = ai_candidates[0]

# --- Coerce numeric and filter ---
df[ai_col] = pd.to_numeric(df[ai_col], errors="coerce")
dfv = df[[ "Company", ai_col ]].dropna().copy()
dfv.rename(columns={ai_col: "AI_Adoption_Index"}, inplace=True)

# --- Distribution: Histogram with hover ---
st.subheader("분포(Distribution)")
st.markdown("<span style='font-size:12px;color:#6b7280'>Histogram of AI Adoption scores; hover to see company & value.</span>", unsafe_allow_html=True)
bins = st.slider("구간 수 (bins)", min_value=5, max_value=30, value=12, step=1)
fig = px.histogram(
    dfv,
    x="AI_Adoption_Index",
    nbins=bins,
    hover_data={"Company": True, "AI_Adoption_Index": ":.1f"},
    opacity=0.85
)
fig.update_traces(marker_line_width=0.5)
fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=420)
st.plotly_chart(fig, use_container_width=True)

# --- Company strip plot with hover ---
st.subheader("기업별 분포 시각화(Company-level view)")
st.markdown("<span style='font-size:12px;color:#6b7280'>Strip plot sorted by AI index; hover to see company & value.</span>", unsafe_allow_html=True)
fig2 = px.strip(
    dfv.sort_values("AI_Adoption_Index"),
    x="AI_Adoption_Index",
    y=["Company"]*len(dfv),
    hover_data={"Company": True, "AI_Adoption_Index": ":.1f"},
    orientation="h"
)
fig2.update_layout(yaxis={"showticklabels": False}, margin=dict(l=10,r=10,t=10,b=10), height=500)
st.plotly_chart(fig2, use_container_width=True)

# --- Summary stats ---
st.subheader("요약 통계(Summary stats)")
desc = dfv["AI_Adoption_Index"].describe()
left, right = st.columns(2)
with left:
    st.metric("유효 표본 수", int(desc["count"]))
    st.metric("평균", f"{desc['mean']:.2f}")
    st.metric("중앙값", f"{desc['50%']:.2f}")
with right:
    st.metric("표준편차", f"{desc['std']:.2f}")
    st.metric("최솟값", f"{desc['min']:.2f}")
    st.metric("최댓값", f"{desc['max']:.2f}")
st.markdown("<span style='font-size:12px;color:#6b7280'>N (valid), Mean, Median, Std, Min, Max.</span>", unsafe_allow_html=True)

# --- Top & Bottom cohorts ---
st.subheader("상위·하위 그룹(Top & Bottom cohorts)")
top_n = st.number_input("상·하위 표시 개수", min_value=3, max_value=15, value=10, step=1)
top = dfv.sort_values("AI_Adoption_Index", ascending=False).head(top_n)
bot = dfv.sort_values("AI_Adoption_Index", ascending=True).head(top_n)

st.write("**상위 기업(Top companies)**")
st.dataframe(top.reset_index(drop=True))
st.write("**하위 기업(Bottom companies)**")
st.dataframe(bot.reset_index(drop=True))

# --- Narrative below charts ---
st.markdown("### 데이터가 보여주는 바 (What the data shows)")
mean = desc['mean']; median = desc['50%']; p75 = dfv["AI_Adoption_Index"].quantile(0.75); p25 = dfv["AI_Adoption_Index"].quantile(0.25)
st.markdown(
    f"- 중심 경향은 **평균 {mean:.2f} / 중앙값 {median:.2f}** 수준이며, 사분위 범위(IQR)는 **{p25:.2f}–{p75:.2f}** 구간입니다.\n"
    f"- **상위 {top_n}** 기업은 우측 꼬리에서 군집하고, **하위 {top_n}** 기업은 좌측 꼬리에 분포합니다."
)
st.markdown("<span style='font-size:12px;color:#6b7280'>Central tendency around mean/median; IQR range. Top cohort clusters near right tail; bottom cohort near left tail.</span>", unsafe_allow_html=True)

# --- Keywords ---
st.markdown("### 키워드 (Keywords)")
st.write(", ".join([
    "AI 도입", "모델/스택 공개", "Responsible AI", "오픈소스 참여",
    "엔지니어링 블로그 발행", "엔터프라이즈 적합성", "거버넌스", "디자인옵스"
]))
st.markdown("<span style='font-size:12px;color:#6b7280'>AI adoption, model disclosure, responsible AI, OSS participation, engineering blog cadence, enterprise readiness, governance, design ops.</span>", unsafe_allow_html=True)

# --- Insights ---
st.markdown("### 인사이트 (Insights)")
st.markdown(
    "1) **공개성과 성숙도는 동행.** 모델/스택 공개와 엔지니어링 콘텐츠 발행이 상위 그룹에서 두드러집니다.\n"
    "2) **거버넌스가 차별점.** ISO·SecSDLC·실험(Analytics) 시그널은 높은 도입도와 함께 나타납니다.\n"
    "3) **GenAI로 디자인옵스 가속.** GenAI 도입은 설계-전달 주기의 효율화와 연계되는 경향이 있습니다."
)
st.markdown("<span style='font-size:12px;color:#6b7280'>1) Disclosure & maturity move together. 2) Governance differentiates enterprise readiness. 3) GenAI accelerates DesignOps.</span>", unsafe_allow_html=True)

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
