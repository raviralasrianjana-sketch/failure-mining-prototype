import os
import datetime as dt

import pandas as pd
import plotly.express as px
import streamlit as st

import auth
import chatbot
from pipeline import run_pipeline, build_trend_table, build_insights, build_word_report

st.set_page_config(
    page_title="Field Returns Failure Mode Mining",
    page_icon="🔧",
    layout="wide",
)

auth.init_db()

# Handle the browser bouncing back from Google sign-in or a "reset password"
# email link (Supabase appends ?code=... to the URL in both cases).
_redirect_result = auth.handle_auth_redirect()
if _redirect_result == "recovery":
    st.session_state.page = "reset_password"
elif _redirect_result == "login":
    st.session_state.page = "upload"

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Cache the pipeline so it doesn't re-run on every single UI interaction
# ---------------------------------------------------------------------------
@st.cache_data
def load_and_process(data_path, filename, n_clusters):
    df, cluster_labels, k_used = run_pipeline(data_path, filename=filename, n_clusters=n_clusters)
    trend = build_trend_table(df)
    return df, cluster_labels, k_used, trend


# ---------------------------------------------------------------------------
# NAVIGATION: the 6 "slides" the user clicks between once analysis is done.
# Each has an icon (animated on the hub page) + a topic name.
# ---------------------------------------------------------------------------
SECTIONS = [
    {"key": "results_overview", "icon": "📊", "label": "Failure Mode Analysis Results"},
    {"key": "theme_overview", "icon": "🧩", "label": "Failure Themes (Overall)"},
    {"key": "trend", "icon": "📈", "label": "Trend Over Time by Theme"},
    {"key": "model_counts", "icon": "🏭", "label": "Failure Counts by Product Model"},
    {"key": "insights", "icon": "💡", "label": "Actionable Insights Summary"},
    {"key": "data_report", "icon": "📄", "label": "Underlying Data & Report (Download)"},
    {"key": "ai_assistant", "icon": "🤖", "label": "Failure Mining AI Assistant (Chat)"},
]
SECTION_KEYS = {s["key"] for s in SECTIONS}


def inject_icon_css():
    st.markdown(
        """
        <style>
        @keyframes pulseIcon {
            0%   { transform: scale(1); }
            50%  { transform: scale(1.18); }
            100% { transform: scale(1); }
        }
        .icon-anim {
            font-size: 46px;
            text-align: center;
            animation: pulseIcon 2.2s ease-in-out infinite;
            padding-top: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def go(page_key):
    st.session_state.page = page_key
    st.rerun()


# ---------------------------------------------------------------------------
# SESSION STATE DEFAULTS
# ---------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "auth"
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


def do_logout():
    auth.logout_user()
    for key in ("user", "access_token", "refresh_token", "df", "cluster_labels", "k_used",
                "trend", "analysis_done", "source_name", "detection_info", "chat_history"):
        st.session_state.pop(key, None)
    st.session_state.page = "auth"
    st.rerun()


# ---------------------------------------------------------------------------
# SLIDE 0 - LOGIN / SIGN UP
# ---------------------------------------------------------------------------
def render_auth_page():
    st.title("🔧 Field Returns Failure Mode Mining")
    st.caption("Sign in to continue -- a profile is created automatically the first time.")

    if not auth.is_configured():
        st.error(
            "Supabase Auth isn't configured yet. Add `SUPABASE_URL` and "
            "`SUPABASE_ANON_KEY` to `.streamlit/secrets.toml` -- see "
            "README.md -> 'Setting up Supabase Auth'."
        )
        return

    # --- Continue with Google -------------------------------------------
    if st.button("🔵 Continue with Google", use_container_width=True):
        try:
            google_url = auth.get_google_login_url()
            st.link_button("Click to continue to Google →", google_url, use_container_width=True)
            st.caption("(Streamlit can't auto-redirect -- tap the link above.)")
        except Exception as e:
            st.error(f"Google sign-in isn't set up yet: {e}")

    st.divider()

    tab_login, tab_signup = st.tabs(["🔑 Log In", "🆕 Sign Up"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log In", type="primary")
        if submitted:
            user, msg = auth.login_user(email, password)
            if user:
                st.session_state.user = user
                st.session_state.page = "upload"
                st.rerun()
            else:
                st.error(msg)

        with st.expander("Forgot password?"):
            with st.form("forgot_password_form"):
                reset_email = st.text_input("Enter your account email", key="reset_email")
                reset_submitted = st.form_submit_button("Send reset link")
            if reset_submitted:
                ok, msg = auth.request_password_reset(reset_email)
                (st.success if ok else st.error)(msg)

    with tab_signup:
        st.caption("Creates your account with Supabase -- use a real email, you'll need it to confirm/recover.")
        with st.form("signup_form"):
            name = st.text_input("Full name", key="signup_name")
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password (min 6 characters)", type="password", key="signup_password")
            submitted = st.form_submit_button("Create Account", type="primary")
        if submitted:
            ok, msg = auth.signup_user(email, password, name)
            if ok:
                if "user" in st.session_state:
                    st.session_state.page = "upload"
                    st.rerun()
                else:
                    st.success(msg)
            else:
                st.error(msg)


def render_reset_password_page():
    st.title("🔑 Choose a new password")
    st.caption("You've arrived here from a password-reset email link.")
    with st.form("reset_password_form"):
        new_password = st.text_input("New password (min 6 characters)", type="password", key="new_password")
        confirm_password = st.text_input("Confirm new password", type="password", key="confirm_password")
        submitted = st.form_submit_button("Update Password", type="primary")
    if submitted:
        if new_password != confirm_password:
            st.error("Passwords don't match.")
        else:
            ok, msg = auth.update_password(new_password)
            if ok:
                st.success(msg)
                if st.button("Go to Log In"):
                    st.session_state.page = "auth"
                    st.rerun()
            else:
                st.error(msg)


# ---------------------------------------------------------------------------
# SIDEBAR: profile card + history + logout (shown whenever logged in)
# ---------------------------------------------------------------------------
def render_sidebar_profile():
    user = st.session_state.user
    st.sidebar.markdown("### 👤 Profile")
    st.sidebar.write(f"**{user['name']}**")
    st.sidebar.caption(user["email"])
    st.sidebar.caption(
        "Signed in via " + ("Google" if user["provider"] == "google" else "Email & Password")
    )

    c1, c2 = st.sidebar.columns(2)
    if c1.button("🕒 History", use_container_width=True):
        go("history")
    if c2.button("🚪 Log Out", use_container_width=True):
        do_logout()

    st.sidebar.divider()


# ---------------------------------------------------------------------------
# SLIDE 1 - FILE UPLOAD
# ---------------------------------------------------------------------------
def render_upload_page():
    st.title("🔧 Field Returns Failure Mode Mining")
    st.caption(
        "Upload your service notes (CSV, Excel, PDF, Word, or an image/"
        "screenshot) and click Analyze to start the failure mode analysis."
    )

    uploaded_file = st.file_uploader(
        "📂 Upload your service notes file",
        type=["csv", "xlsx", "xls", "tsv", "json", "pdf", "docx", "png", "jpg", "jpeg"],
        help="Accepts CSV, Excel (.xlsx/.xls), TSV, JSON, PDF, Word (.docx), "
             "or images (.png/.jpg/.jpeg -- text is read via OCR). Spreadsheet "
             "files must include at least: product_model, date, symptom_text "
             "columns. serial_range/serial_number and tag columns are optional "
             "but used if present. PDFs/Word docs/images are read as free text "
             "and parsed automatically.",
    )
    st.caption(
        "Supported input:  ✓ Structured Failure Mining datasets  "
        "✓ Raw customer/review data (CSV/XLSX, e.g. a Google Reviews export) "
        "-- the review/comment column is detected automatically."
    )

    auto_k = st.sidebar.checkbox("Auto-select number of themes", value=True)
    n_clusters = None
    if not auto_k:
        n_clusters = st.sidebar.slider("Number of themes (clusters)", 3, 12, 8)

    if uploaded_file is None:
        st.info("Please upload a file (CSV, Excel, PDF, Word, or image).")
        return

    st.success("File uploaded successfully.")
    analyze = st.button("🔍 Analyze", type="primary")

    if not analyze:
        return

    import tempfile
    import traceback

    _, real_ext = os.path.splitext(uploaded_file.name)
    if not real_ext:
        real_ext = ".csv"

    with tempfile.NamedTemporaryFile(delete=False, suffix=real_ext) as tmp:
        tmp.write(uploaded_file.getvalue())
        data_path = tmp.name

    try:
        with st.spinner("Running pipeline: sanitizing → vectorizing → clustering → labeling..."):
            df, cluster_labels, k_used, trend = load_and_process(
                data_path, uploaded_file.name, n_clusters
            )

        st.session_state.df = df
        st.session_state.cluster_labels = cluster_labels
        st.session_state.k_used = k_used
        st.session_state.trend = trend
        st.session_state.source_name = uploaded_file.name
        st.session_state.detection_info = df.attrs.get("detection_info", {"input_type": "structured"})
        st.session_state.analysis_done = True

        # --- Save a snapshot report + log this run to the profile's history ---
        top_theme = df["theme"].value_counts().idxmax() if len(df) else "N/A"
        insights_full = build_insights(df, top_n=k_used)
        report_bytes = build_word_report(
            df, trend, insights_full, k_used, source_name=uploaded_file.name
        )
        safe_email = st.session_state.user["email"].replace("@", "_at_").replace(".", "_")
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(REPORTS_DIR, f"{safe_email}_{timestamp}.docx")
        with open(report_path, "wb") as f:
            f.write(report_bytes)

        auth.add_history(
            user_email=st.session_state.user["email"],
            source_name=uploaded_file.name,
            total_notes=len(df),
            k_used=k_used,
            top_theme=top_theme,
            report_path=report_path,
        )

        go("hub")

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Couldn't process this file: {e}")
        with st.expander("Full error details"):
            st.code(traceback.format_exc())


# ---------------------------------------------------------------------------
# Shared helper: apply sidebar filters, used by the hub + every section page
# ---------------------------------------------------------------------------
def get_filtered_data():
    df = st.session_state.df
    trend = st.session_state.trend

    models = sorted(df["product_model"].unique())
    selected_models = st.sidebar.multiselect("Filter by product model", models, default=models)

    date_min, date_max = df["date"].min(), df["date"].max()
    date_range = st.sidebar.date_input(
        "Filter by date range", value=(date_min.date(), date_max.date())
    )

    filtered = df[df["product_model"].isin(selected_models)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        filtered = filtered[(filtered["date"] >= start) & (filtered["date"] <= end)]

    trend_f = trend[trend["product_model"].isin(selected_models)]
    trend_f = trend_f.groupby(["month", "theme"])["count"].sum().reset_index()

    return filtered, trend_f, selected_models


def back_button():
    if st.button("⬅️ Back to Hub"):
        go("hub")


# ---------------------------------------------------------------------------
# HUB - icon grid, one entry per section. Click the icon+name to open it.
# ---------------------------------------------------------------------------
def render_hub():
    df = st.session_state.df
    k_used = st.session_state.k_used

    st.title("📊 Failure Mode Analysis Results")
    st.caption(
        f"Source: {st.session_state.source_name}  ·  {len(df)} notes  ·  "
        f"{k_used} failure themes discovered. Click a section below to open it."
    )

    detection_info = st.session_state.get("detection_info", {"input_type": "structured"})
    if detection_info.get("input_type") == "raw_reviews":
        ai_note = (
            "AI-enhanced extraction" if detection_info.get("ai_extraction_used")
            else "free keyword-based extraction (no AI API configured)"
        )
        st.info(
            f"🔎 Detected input type: **Raw Review Data**  ·  "
            f"Review column detected: **{detection_info.get('review_column')}**  ·  "
            f"{detection_info.get('rows_dropped', 0)} empty/duplicate rows skipped  ·  "
            f"Component/severity extracted via {ai_note}."
        )

    inject_icon_css()

    for sec in SECTIONS:
        with st.container(border=True):
            c1, c2 = st.columns([1, 6])
            with c1:
                st.markdown(f'<div class="icon-anim">{sec["icon"]}</div>', unsafe_allow_html=True)
            with c2:
                st.write("")
                if st.button(sec["label"], key=f"nav_{sec['key']}", use_container_width=True):
                    go(sec["key"])

    if st.button("⬅️ Analyze a different file"):
        st.session_state.analysis_done = False
        go("upload")


# ---------------------------------------------------------------------------
# SECTION 1 - Failure Mode Analysis Results (KPI overview)
# ---------------------------------------------------------------------------
def render_results_overview():
    filtered, trend_f, selected_models = get_filtered_data()
    k_used = st.session_state.k_used

    back_button()
    st.title("📊 Failure Mode Analysis Results")
    st.caption(
        "Service/repair notes → clustered themes → trends → actionable insights. "
        "All data below is synthetic (no real customer information)."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Notes (filtered)", len(filtered))
    col2.metric("Themes Discovered", k_used)
    top_theme = filtered["theme"].value_counts().idxmax() if len(filtered) else "N/A"
    col3.metric("Top Failure Mode", top_theme)
    col4.metric("Product Models", filtered["product_model"].nunique())

    st.divider()
    st.caption(
        "Use the icons on the Hub to drill into themes, trends, model "
        "breakdowns, insights, and the downloadable report."
    )


# ---------------------------------------------------------------------------
# SECTION 2 - Failure Themes (Overall)
# ---------------------------------------------------------------------------
def render_theme_overview():
    filtered, _, _ = get_filtered_data()

    back_button()
    st.title("🧩 Failure Themes (Overall)")

    theme_counts = filtered["theme"].value_counts().reset_index()
    theme_counts.columns = ["theme", "count"]

    fig_bar = px.bar(
        theme_counts.sort_values("count"),
        x="count", y="theme", orientation="h",
        color="count", color_continuous_scale="Blues",
    )
    fig_bar.update_layout(showlegend=False, coloraxis_showscale=False, height=480)
    st.plotly_chart(fig_bar, use_container_width=True)


# ---------------------------------------------------------------------------
# SECTION 3 - Trend Over Time by Theme
# ---------------------------------------------------------------------------
def render_trend():
    _, trend_f, _ = get_filtered_data()

    back_button()
    st.title("📈 Trend Over Time by Theme")

    fig_line = px.line(trend_f, x="month", y="count", color="theme", markers=True)
    fig_line.update_layout(height=480, legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig_line, use_container_width=True)


# ---------------------------------------------------------------------------
# SECTION 4 - Failure Counts by Product Model
# ---------------------------------------------------------------------------
def render_model_counts():
    filtered, _, _ = get_filtered_data()

    back_button()
    st.title("🏭 Failure Counts by Product Model")

    model_theme = filtered.groupby(["product_model", "theme"]).size().reset_index(name="count")
    fig_stack = px.bar(model_theme, x="product_model", y="count", color="theme", barmode="stack")
    fig_stack.update_layout(height=450)
    st.plotly_chart(fig_stack, use_container_width=True)


# ---------------------------------------------------------------------------
# SECTION 5 - Actionable Insights Summary
# ---------------------------------------------------------------------------
def render_insights():
    filtered, _, _ = get_filtered_data()
    k_used = st.session_state.k_used

    back_button()
    st.title("💡 Actionable Insights Summary")

    insights = build_insights(filtered, top_n=k_used)

    for ins in insights:
        header = (
            f"**{ins['theme']}** — {ins['count']} notes ({ins['pct_of_total']}% of total) · "
            f"most affected: {ins['most_affected_model']} · "
            f"serial range: {ins['most_common_serial_range']}"
        )
        with st.expander(header):
            if ins.get("tag_agreement_pct", 0) > 0:
                st.caption(
                    f"🏷️ Structured-tag agreement: {ins['tag_agreement_pct']}% "
                    "(how often the technician's own tag matched this theme)"
                )
            st.markdown("**Example notes (sanitized):**")
            for ex in ins["examples"]:
                st.markdown(f"- {ex}")


# ---------------------------------------------------------------------------
# SECTION 6 - Underlying Data & Report (download)
# ---------------------------------------------------------------------------
def render_data_report():
    filtered, trend_f, selected_models = get_filtered_data()
    k_used = st.session_state.k_used
    trend = st.session_state.trend

    back_button()
    st.title("📄 Underlying Data & Report")

    st.subheader("Underlying Data (sanitized)")
    display_cols = [
        "note_id", "product_model", "serial_range", "date", "theme", "tag",
        "symptom_text_clean", "fix_text_clean",
    ]
    # component/severity only exist when the input was raw review data
    # (see review_preprocessor.py) -- shown only when present, so
    # structured datasets (which never have these columns) look exactly
    # as before.
    for extra_col in ("component", "severity"):
        if extra_col in filtered.columns:
            display_cols.append(extra_col)
    st.dataframe(filtered[display_cols], use_container_width=True, height=320)

    st.divider()
    st.subheader("⬇️ Download Full Report")
    st.caption(
        "One Word document with the summary, all charts, and the actionable "
        "insights for the currently filtered view -- ready to share or submit."
    )

    insights = build_insights(filtered, top_n=k_used)
    with st.spinner("Building your report..."):
        report_bytes = build_word_report(
            filtered,
            trend[trend["product_model"].isin(selected_models)],
            insights,
            k_used,
            source_name=st.session_state.get("source_name", "uploaded file"),
        )

    st.download_button(
        label="📄 Download Analysis Report (Word)",
        data=report_bytes,
        file_name="failure_mode_analysis_report.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        type="primary",
    )

    st.caption(
        "A snapshot of the full (unfiltered) report was also saved to your "
        "profile's History when you first ran this analysis."
    )


# ---------------------------------------------------------------------------
# SECTION 7 - Failure Mining AI Assistant (Groq-powered chat)
# ---------------------------------------------------------------------------
def render_ai_assistant():
    filtered, trend_f, selected_models = get_filtered_data()
    k_used = st.session_state.k_used

    back_button()
    st.title("🤖 Failure Mining AI Assistant")
    st.caption(
        "Ask questions about the current analysis results (themes, trends, "
        "affected models). Answers are grounded only in this analysis -- "
        "powered by Groq."
    )

    if not chatbot.is_configured():
        st.warning(
            "The AI Assistant isn't configured yet -- set `GROQ_API_KEY` "
            "(as an environment variable or in `.streamlit/secrets.toml`). "
            "See README.md -> 'Setting up the AI Assistant'."
        )
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    insights = build_insights(filtered, top_n=k_used)
    context = chatbot.build_context(
        filtered, insights, trend_f, k_used,
        source_name=st.session_state.get("source_name", "uploaded file"),
    )

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    question = st.chat_input("Ask about the failure themes, trends, or affected models...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = chatbot.ask_assistant(
                        context, st.session_state.chat_history[:-1], question
                    )
                except RuntimeError as e:
                    answer = f"⚠️ {e}"
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.rerun()


# ---------------------------------------------------------------------------
# HISTORY - past analyses for the logged-in profile
# ---------------------------------------------------------------------------
def render_history():
    user = st.session_state.user

    if st.session_state.analysis_done:
        if st.button("⬅️ Back to Hub"):
            go("hub")
    else:
        if st.button("⬅️ Back to Upload"):
            go("upload")

    st.title("🕒 Analysis History")
    st.caption(f"Past analyses run by {user['name']} ({user['email']}).")

    rows = auth.get_history(user["email"])

    if not rows:
        st.info("No analyses yet -- upload a file to run your first one.")
        return

    for row in rows:
        when = row["created_at"].split(".")[0].replace("T", " ")
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"**{row['source_name']}** · {when}")
                st.caption(
                    f"{row['total_notes']} notes · {row['k_used']} themes · "
                    f"top failure mode: {row['top_theme']}"
                )
            with c2:
                report_path = row.get("report_path")
                if report_path and os.path.exists(report_path):
                    with open(report_path, "rb") as f:
                        st.download_button(
                            "📄 Report",
                            data=f.read(),
                            file_name=os.path.basename(report_path),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key=f"dl_{row['id']}",
                        )
                else:
                    st.caption("Report unavailable")


# ---------------------------------------------------------------------------
# ROUTER
# ---------------------------------------------------------------------------
if st.session_state.page == "reset_password":
    render_reset_password_page()
elif "user" not in st.session_state:
    render_auth_page()
else:
    st.sidebar.title("🔧 Controls")
    render_sidebar_profile()

    page = st.session_state.page

    if page == "history":
        render_history()
    elif not st.session_state.analysis_done:
        render_upload_page()
    elif page == "hub" or page not in SECTION_KEYS:
        render_hub()
    elif page == "results_overview":
        render_results_overview()
    elif page == "theme_overview":
        render_theme_overview()
    elif page == "trend":
        render_trend()
    elif page == "model_counts":
        render_model_counts()
    elif page == "insights":
        render_insights()
    elif page == "data_report":
        render_data_report()
    elif page == "ai_assistant":
        render_ai_assistant()
    else:
        render_hub()

    st.sidebar.caption(
        "MVP scope only — production hardening (robust PII detection, "
        "human-in-the-loop labeling, live data ingestion) not included."
    )