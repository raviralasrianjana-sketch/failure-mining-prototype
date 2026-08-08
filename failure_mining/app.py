"""
app.py
------
The Streamlit dashboard. This file is ONLY responsible for layout/UI --
all the actual data science logic lives in pipeline.py, which keeps
things easy to read and easy to debug separately.

Run with:  streamlit run app.py
"""

import io
import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline import run_pipeline, build_trend_table, build_insights

st.set_page_config(
    page_title="Field Returns Failure Mode Mining",
    page_icon="🔧",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cache the pipeline so it doesn't re-run on every single UI interaction
# (Streamlit re-runs the whole script top-to-bottom on every click!)
# ---------------------------------------------------------------------------
@st.cache_data
def load_and_process(csv_path, n_clusters):
    df, cluster_labels, k_used = run_pipeline(csv_path, n_clusters=n_clusters)
    trend = build_trend_table(df)
    return df, cluster_labels, k_used, trend


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("🔧 Controls")
st.sidebar.markdown(
    "This dashboard clusters field-return service notes into failure "
    "themes automatically -- no manual tagging required."
)

uploaded_file = st.file_uploader(
    "📂 Upload your CSV file",
    type=["csv"]
)

auto_k = st.sidebar.checkbox("Auto-select number of themes", value=True)
n_clusters = None
if not auto_k:
    n_clusters = st.sidebar.slider("Number of themes (clusters)", 3, 12, 8)

analyze = st.button("🔍 Analyze", type="primary")

if uploaded_file is None:
    st.info("Please upload a CSV file and click Analyze.")
    st.stop()

if not analyze:
    st.info("File uploaded successfully. Click Analyze to start processing.")
    st.stop()

import tempfile

with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
    tmp.write(uploaded_file.getvalue())
    data_path = tmp.name

with st.spinner("Running pipeline: sanitizing → vectorizing → clustering → labeling..."):
    df, cluster_labels, k_used, trend = load_and_process(data_path, n_clusters)

st.sidebar.success(f"Discovered {k_used} failure themes from {len(df)} notes.")

models = sorted(df["product_model"].unique())
selected_models = st.sidebar.multiselect("Filter by product model", models, default=models)

date_min, date_max = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input(
    "Filter by date range", value=(date_min.date(), date_max.date())
)

# Apply filters
filtered = df[df["product_model"].isin(selected_models)]
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    filtered = filtered[(filtered["date"] >= start) & (filtered["date"] <= end)]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🔧 Field Returns Failure Mode Mining")
st.caption(
    "Service/repair notes → clustered themes → trends → actionable insights. "
    "All data below is synthetic (no real customer information)."
)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Notes (filtered)", len(filtered))
col2.metric("Themes Discovered", k_used)
top_theme = filtered["theme"].value_counts().idxmax() if len(filtered) else "N/A"
col3.metric("Top Failure Mode", top_theme)
col4.metric("Product Models", filtered["product_model"].nunique())

st.divider()

# ---------------------------------------------------------------------------
# Theme distribution + Trend over time
# ---------------------------------------------------------------------------
left, right = st.columns([1, 1.3])

with left:
    st.subheader("Failure Themes (Overall)")
    theme_counts = (
        filtered["theme"].value_counts().reset_index()
    )
    theme_counts.columns = ["theme", "count"]
    fig_bar = px.bar(
        theme_counts.sort_values("count"),
        x="count", y="theme", orientation="h",
        color="count", color_continuous_scale="Blues",
    )
    fig_bar.update_layout(showlegend=False, coloraxis_showscale=False, height=420)
    st.plotly_chart(fig_bar, use_container_width=True)

with right:
    st.subheader("Trend Over Time by Theme")
    trend_f = trend[trend["product_model"].isin(selected_models)]
    trend_f = trend_f.groupby(["month", "theme"])["count"].sum().reset_index()
    fig_line = px.line(
        trend_f, x="month", y="count", color="theme", markers=True,
    )
    fig_line.update_layout(height=420, legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig_line, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Trend by product model (stacked)
# ---------------------------------------------------------------------------
st.subheader("Failure Counts by Product Model")
model_theme = filtered.groupby(["product_model", "theme"]).size().reset_index(name="count")
fig_stack = px.bar(
    model_theme, x="product_model", y="count", color="theme", barmode="stack"
)
fig_stack.update_layout(height=400)
st.plotly_chart(fig_stack, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Actionable insights summary
# ---------------------------------------------------------------------------
st.subheader("📋 Actionable Insights Summary")
insights = build_insights(filtered, top_n=k_used)

for ins in insights:
    with st.expander(
        f"**{ins['theme']}** — {ins['count']} notes "
        f"({ins['pct_of_total']}% of total) · most affected: {ins['most_affected_model']}"
    ):
        st.markdown("**Example notes (sanitized):**")
        for ex in ins["examples"]:
            st.markdown(f"- {ex}")

st.divider()

# ---------------------------------------------------------------------------
# Raw data table + export
# ---------------------------------------------------------------------------
st.subheader("📄 Underlying Data (sanitized)")
display_cols = [
    "note_id", "product_model", "date", "theme",
    "symptom_text_clean", "fix_text_clean",
]
st.dataframe(filtered[display_cols], use_container_width=True, height=300)

# Export report as CSV
export_df = filtered[display_cols]
csv_buffer = io.StringIO()
export_df.to_csv(csv_buffer, index=False)
st.download_button(
    label="⬇️ Download filtered report as CSV",
    data=csv_buffer.getvalue(),
    file_name="failure_mode_report.csv",
    mime="text/csv",
)

st.caption(
    "MVP scope only — production hardening (robust PII detection, "
    "human-in-the-loop labeling, live data ingestion) not included."
)
