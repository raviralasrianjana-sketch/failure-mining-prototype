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
# ---------------------------------------------------------------------------
@st.cache_data
def load_and_process(csv_path, n_clusters):
    df, cluster_labels, k_used = run_pipeline(csv_path, n_clusters=n_clusters)
    trend = build_trend_table(df)
    return df, cluster_labels, k_used, trend


# ---------------------------------------------------------------------------
# SESSION STATE - controls which slide/page is shown
# ---------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "upload"

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# ---------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------------------------
st.sidebar.title("🔧 Controls")
st.sidebar.markdown(
    "This dashboard clusters field-return service notes into failure "
    "themes automatically -- no manual tagging required."
)


# ---------------------------------------------------------------------------
# SLIDE 1 - FILE UPLOAD
# ---------------------------------------------------------------------------
if st.session_state.page == "upload":

    st.title("🔧 Field Returns Failure Mode Mining")
    st.caption(
        "Upload your CSV file and click Analyze to start the failure mode analysis."
    )

    uploaded_file = st.file_uploader(
        "📂 Upload your CSV file",
        type=["csv"]
    )

    auto_k = st.sidebar.checkbox("Auto-select number of themes", value=True)

    n_clusters = None

    if not auto_k:
        n_clusters = st.sidebar.slider(
            "Number of themes (clusters)",
            3,
            12,
            8
        )

    if uploaded_file is None:
        st.info("Please upload a CSV file.")

    else:
        st.success("File uploaded successfully.")

        analyze = st.button("🔍 Analyze", type="primary")

        if analyze:

            import tempfile

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".csv"
            ) as tmp:

                tmp.write(uploaded_file.getvalue())
                data_path = tmp.name

            with st.spinner(
                "Running pipeline: sanitizing → vectorizing → clustering → labeling..."
            ):

                df, cluster_labels, k_used, trend = load_and_process(
                    data_path,
                    n_clusters
                )

            # Store the analysis results
            st.session_state.df = df
            st.session_state.cluster_labels = cluster_labels
            st.session_state.k_used = k_used
            st.session_state.trend = trend

            st.session_state.analysis_done = True

            # Move to results slide
            st.session_state.page = "results"

            st.rerun()


# ---------------------------------------------------------------------------
# SLIDE 2 - ANALYSIS RESULTS
# ---------------------------------------------------------------------------
else:

    df = st.session_state.df
    k_used = st.session_state.k_used
    trend = st.session_state.trend

    # Back button
    if st.button("⬅️ Back to Upload"):

        st.session_state.page = "upload"
        st.session_state.analysis_done = False

        st.rerun()

    st.sidebar.success(
        f"Discovered {k_used} failure themes from {len(df)} notes."
    )

    models = sorted(df["product_model"].unique())

    selected_models = st.sidebar.multiselect(
        "Filter by product model",
        models,
        default=models
    )

    date_min, date_max = df["date"].min(), df["date"].max()

    date_range = st.sidebar.date_input(
        "Filter by date range",
        value=(date_min.date(), date_max.date())
    )

    # Apply filters
    filtered = df[df["product_model"].isin(selected_models)]

    if isinstance(date_range, tuple) and len(date_range) == 2:

        start, end = (
            pd.Timestamp(date_range[0]),
            pd.Timestamp(date_range[1])
        )

        filtered = filtered[
            (filtered["date"] >= start) &
            (filtered["date"] <= end)
        ]


    # -----------------------------------------------------------------------
    # HEADER
    # -----------------------------------------------------------------------
    st.title("📊 Failure Mode Analysis Results")

    st.caption(
        "Service/repair notes → clustered themes → trends → actionable insights. "
        "All data below is synthetic (no real customer information)."
    )


    # -----------------------------------------------------------------------
    # KPI ROW
    # -----------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Notes (filtered)",
        len(filtered)
    )

    col2.metric(
        "Themes Discovered",
        k_used
    )

    top_theme = (
        filtered["theme"].value_counts().idxmax()
        if len(filtered)
        else "N/A"
    )

    col3.metric(
        "Top Failure Mode",
        top_theme
    )

    col4.metric(
        "Product Models",
        filtered["product_model"].nunique()
    )

    st.divider()


    # -----------------------------------------------------------------------
    # THEME DISTRIBUTION + TREND OVER TIME
    # -----------------------------------------------------------------------
    left, right = st.columns([1, 1.3])

    with left:

        st.subheader("Failure Themes (Overall)")

        theme_counts = (
            filtered["theme"]
            .value_counts()
            .reset_index()
        )

        theme_counts.columns = ["theme", "count"]

        fig_bar = px.bar(
            theme_counts.sort_values("count"),
            x="count",
            y="theme",
            orientation="h",
            color="count",
            color_continuous_scale="Blues",
        )

        fig_bar.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            height=420
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )


    with right:

        st.subheader("Trend Over Time by Theme")

        trend_f = trend[
            trend["product_model"].isin(selected_models)
        ]

        trend_f = (
            trend_f
            .groupby(["month", "theme"])["count"]
            .sum()
            .reset_index()
        )

        fig_line = px.line(
            trend_f,
            x="month",
            y="count",
            color="theme",
            markers=True,
        )

        fig_line.update_layout(
            height=420,
            legend=dict(
                orientation="h",
                y=-0.3
            )
        )

        st.plotly_chart(
            fig_line,
            use_container_width=True
        )


    st.divider()


    # -----------------------------------------------------------------------
    # TREND BY PRODUCT MODEL
    # -----------------------------------------------------------------------
    st.subheader("Failure Counts by Product Model")

    model_theme = (
        filtered
        .groupby(["product_model", "theme"])
        .size()
        .reset_index(name="count")
    )

    fig_stack = px.bar(
        model_theme,
        x="product_model",
        y="count",
        color="theme",
        barmode="stack"
    )

    fig_stack.update_layout(height=400)

    st.plotly_chart(
        fig_stack,
        use_container_width=True
    )


    st.divider()


    # -----------------------------------------------------------------------
    # ACTIONABLE INSIGHTS SUMMARY
    # -----------------------------------------------------------------------
    st.subheader("📋 Actionable Insights Summary")

    insights = build_insights(
        filtered,
        top_n=k_used
    )

    for ins in insights:

        with st.expander(
            f"**{ins['theme']}** — {ins['count']} notes "
            f"({ins['pct_of_total']}% of total) · "
            f"most affected: {ins['most_affected_model']}"
        ):

            st.markdown("**Example notes (sanitized):**")

            for ex in ins["examples"]:

                st.markdown(
                    f"- {ex}"
                )


    st.divider()


    # -----------------------------------------------------------------------
    # RAW DATA TABLE + EXPORT
    # -----------------------------------------------------------------------
    st.subheader("📄 Underlying Data (sanitized)")

    display_cols = [
        "note_id",
        "product_model",
        "date",
        "theme",
        "symptom_text_clean",
        "fix_text_clean",
    ]

    st.dataframe(
        filtered[display_cols],
        use_container_width=True,
        height=300
    )


    # Export report as CSV
    export_df = filtered[display_cols]

    csv_buffer = io.StringIO()

    export_df.to_csv(
        csv_buffer,
        index=False
    )

    st.download_button(
        label="⬇️ Download filtered report as CSV",
        data=csv_buffer.getvalue(),
        file_name="failure_mode_report.csv",
        mime="text/csv",
    )


    st.divider()

    st.caption(
        "MVP scope only — production hardening (robust PII detection, "
        "human-in-the-loop labeling, live data ingestion) not included."
    )