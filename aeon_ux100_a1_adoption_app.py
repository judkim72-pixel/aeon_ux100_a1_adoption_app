
import os
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="AEON UX100 · A-1 · AI Adoption", layout="wide")
st.title("AEON UX100 · A-1 · AI Adoption Score Distribution")

# --- Auto-load root dataset if present ---
DEFAULT_DATA_PATH = "ux_100_dataset.xlsx"  # repo root filename
df = None
auto_loaded = False

if os.path.exists(DEFAULT_DATA_PATH):
    try:
        df = pd.read_excel(DEFAULT_DATA_PATH, sheet_name="Data")
        auto_loaded = True
        st.success(f"Loaded default dataset: {DEFAULT_DATA_PATH}")
    except Exception as e:
        st.warning(f"Found {DEFAULT_DATA_PATH} but failed to read: {e}")

# --- Fallback UI only when auto-load not available ---
if df is None:
    st.markdown("Upload the consolidated Excel (sheet: **Data**) or provide a path below.")
    uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    path_in = st.text_input("Or type a server path", value="", help="Leave blank if using file uploader.")

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
            st.error(f"Failed to read uploaded file: {e}")
    elif path_in:
        try:
            df = load_df_from_path(path_in)
        except Exception as e:
            st.error(f"Failed to read path: {e}")

if df is None:
    st.info("Waiting for data... (sheet name must be 'Data' and include 'Company' and 'AI Adoption Index (0–5)')")
    st.stop()

# --- Identify AI column ---
ai_candidates = [c for c in df.columns if "AI Adoption Index" in str(c)]
if not ai_candidates:
    st.error("Column 'AI Adoption Index (0–5)' not found in 'Data' sheet.")
    st.stop()
ai_col = ai_candidates[0]

# --- Coerce numeric and filter ---
df[ai_col] = pd.to_numeric(df[ai_col], errors="coerce")
dfv = df[[ "Company", ai_col ]].dropna().copy()
dfv.rename(columns={ai_col: "AI_Adoption_Index"}, inplace=True)

# --- Distribution: Histogram with hover ---
st.subheader("Distribution")
bins = st.slider("Number of bins", min_value=5, max_value=30, value=12, step=1)
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
st.subheader("Company-level view")
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
st.subheader("Summary stats")
desc = dfv["AI_Adoption_Index"].describe()
left, right = st.columns(2)
with left:
    st.metric("N (valid)", int(desc["count"]))
    st.metric("Mean", f"{desc['mean']:.2f}")
    st.metric("Median", f"{desc['50%']:.2f}")
with right:
    st.metric("Std", f"{desc['std']:.2f}")
    st.metric("Min", f"{desc['min']:.2f}")
    st.metric("Max", f"{desc['max']:.2f}")

# --- Top & Bottom cohorts ---
top_n = st.number_input("Top & Bottom N", min_value=3, max_value=15, value=10, step=1)
top = dfv.sort_values("AI_Adoption_Index", ascending=False).head(top_n)
bot = dfv.sort_values("AI_Adoption_Index", ascending=True).head(top_n)

st.write("**Top companies**")
st.dataframe(top.reset_index(drop=True))
st.write("**Bottom companies**")
st.dataframe(bot.reset_index(drop=True))

# --- Narrative below charts ---
st.markdown("### What the data shows")
mean = desc['mean']; median = desc['50%']; p75 = dfv["AI_Adoption_Index"].quantile(0.75); p25 = dfv["AI_Adoption_Index"].quantile(0.25)
st.markdown(
    f"- Central tendency around **mean {mean:.2f} / median {median:.2f}**; IQR **{p25:.2f}–{p75:.2f}**.\n"
    f"- **Top {top_n}** cluster near right tail; **Bottom {top_n}** near left tail."
)

# --- Keywords ---
st.markdown("### Keywords")
st.write(", ".join([
    "AI adoption", "model disclosure", "responsible AI", "OSS participation",
    "engineering blog cadence", "enterprise readiness", "governance", "design ops"
]))

# --- Insights ---
st.markdown("### Insights")
st.markdown(
    "1. **Disclosure and maturity rise together.** Model/stack notes + active engineering content commonly appear in the top cohort.\n"
    "2. **Governance differentiates enterprise readiness.** ISO/SecSDLC/analytics signals co-occur with higher adoption.\n"
    "3. **DesignOps accelerates with GenAI.** When present, it correlates with higher adoption values."
)

# --- Top vs Bottom bullets ---
st.markdown("### Top vs. Bottom (differences)")
def bulletize(dfpart, label):
    names = " · ".join(dfpart["Company"].head(7).tolist())
    return f"- **{label}**: {names}" + (" …" if len(dfpart)>7 else "")
st.markdown(bulletize(top, "Top cohort"))
st.markdown(bulletize(bot, "Bottom cohort"))

if auto_loaded:
    st.caption("Loaded from repo-root dataset (ux_100_dataset.xlsx). Upload UI is hidden when auto-load succeeds.")
else:
    st.caption("Snapshot-only. For trends, capture the same fields monthly/quarterly.")
