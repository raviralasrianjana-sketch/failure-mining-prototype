import os
import datetime as dt

import pandas as pd
import plotly.express as px
import streamlit as st

import auth
import chatbot
from pipeline import run_pipeline_batch, build_insights, build_word_report

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
@st.cache_data(show_spinner=False)
def load_and_process_batch(file_items, n_clusters):
    """Process ALL uploaded files as one dataset.

    The expensive TF-IDF + clustering stage is intentionally run once for the
    combined batch instead of once per uploaded file.
    """
    return run_pipeline_batch(file_items, n_clusters=n_clusters)


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


def inject_auth_theme():
    """Dark navy / teal 'failure mining' theme for the sign-in screen only.

    Restyles Streamlit's own widgets (inputs, buttons, tabs, bordered
    containers) via CSS instead of replacing them with raw HTML, so the
    real st.form / auth.login_user() flow keeps working exactly as before.
    """
    st.markdown(
        """
        <style>
        :root{
            --fm-navy-deep:#081420;
            --fm-teal:#2FE0C4;
            --fm-teal-dim:#1B7F71;
            --fm-teal-glow:rgba(47,224,196,0.55);
            --fm-ink:#EAF4F2;
            --fm-ink-dim:#8FA9AC;
            --fm-line:rgba(143,169,172,0.25);
        }

        /* --- page background: static low-poly network, no JS needed --- */
        [data-testid="stAppViewContainer"], .stApp{
            background-color:var(--fm-navy-deep);
            background-image:
                radial-gradient(ellipse at 20% 15%, #0F2C3F 0%, transparent 55%),
                radial-gradient(ellipse at 85% 90%, #0C2131 0%, transparent 50%),
                url("data:image/svg+xml;utf8,\
<svg xmlns='http://www.w3.org/2000/svg' width='900' height='900'>\
<g fill='none' stroke='%232FE0C4' stroke-opacity='0.14'>\
<polygon points='60,40 220,120 90,220'/>\
<polygon points='700,60 860,150 760,260'/>\
<polygon points='120,600 300,520 260,720'/>\
<polygon points='640,560 820,640 660,780'/>\
<line x1='60' y1='40' x2='700' y2='60'/>\
<line x1='90' y1='220' x2='260' y2='720'/>\
<line x1='760' y1='260' x2='660' y2='780'/>\
</g>\
<g fill='%232FE0C4' fill-opacity='0.5'>\
<circle cx='60' cy='40' r='3'/><circle cx='220' cy='120' r='2.5'/>\
<circle cx='90' cy='220' r='3'/><circle cx='700' cy='60' r='2.5'/>\
<circle cx='860' cy='150' r='3'/><circle cx='760' cy='260' r='2.5'/>\
<circle cx='120' cy='600' r='3'/><circle cx='300' cy='520' r='2.5'/>\
<circle cx='260' cy='720' r='3'/><circle cx='640' cy='560' r='2.5'/>\
<circle cx='820' cy='640' r='3'/><circle cx='660' cy='780' r='2.5'/>\
</g></svg>");
            background-repeat:repeat;
            background-size:900px 900px, 900px 900px, 900px 900px;
        }

        [data-testid="stHeader"]{ background:transparent; }

        /* --- headings / captions on the auth page --- */
        #fm-eyebrow{
            font-family:'Courier New', monospace;
            font-size:1.05rem;
            letter-spacing:0.16em;
            text-transform:uppercase;
            color:var(--fm-teal);
            font-weight:700;
            margin-bottom:0.6rem;
        }
        #fm-title{
            font-size:3.1rem;
            font-weight:750;
            color:var(--fm-ink);
            letter-spacing:-0.01em;
            margin:0 0 0.7rem 0;
            line-height:1.15;
        }
        #fm-tagline{
            font-size:1.4rem;
            font-weight:650;
            color:var(--fm-teal);
            margin:0 0 1.1rem 0;
        }
        #fm-sub{
            font-size:1.25rem;
            color:var(--fm-ink-dim);
            line-height:1.6;
            max-width:560px;
        }
        #fm-points div{
            display:flex;
            align-items:center;
            gap:0.7rem;
            font-size:1.2rem;
            color:var(--fm-ink);
            margin-top:1.3rem;
        }

        /* --- the sign-in card itself: Streamlit's bordered container --- */
        div[data-testid="stVerticalBlockBorderWrapper"]{
            background:linear-gradient(180deg, rgba(14,34,51,0.94), rgba(8,20,32,0.96));
            border:1px solid var(--fm-line) !important;
            border-radius:16px !important;
            box-shadow:0 30px 80px -20px rgba(0,0,0,0.65);
            padding:0.5rem 0.25rem;
        }

        /* text/captions inside the card */
        div[data-testid="stVerticalBlockBorderWrapper"] p,
        div[data-testid="stVerticalBlockBorderWrapper"] label,
        div[data-testid="stVerticalBlockBorderWrapper"] span{
            color:var(--fm-ink) !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stCaptionContainer"]{
            color:var(--fm-ink-dim) !important;
        }

        /* inputs */
        div[data-testid="stVerticalBlockBorderWrapper"] input{
            background:rgba(255,255,255,0.03) !important;
            border:1px solid var(--fm-line) !important;
            border-radius:8px !important;
            color:var(--fm-ink) !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] input:focus{
            border-color:var(--fm-teal-dim) !important;
            box-shadow:0 0 0 3px rgba(47,224,196,0.15) !important;
        }

        /* tabs (Log In / Sign Up) */
        div[data-testid="stVerticalBlockBorderWrapper"] [data-baseweb="tab-list"]{
            background:rgba(255,255,255,0.03);
            border-radius:8px;
            gap:0;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] [data-baseweb="tab"]{
            color:var(--fm-ink-dim) !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] [aria-selected="true"]{
            color:var(--fm-teal) !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] [data-baseweb="tab-highlight"]{
            background-color:var(--fm-teal) !important;
        }

        /* primary buttons (Log In / Create Account / form submits) */
        div[data-testid="stVerticalBlockBorderWrapper"] button[kind="primary"],
        div[data-testid="stVerticalBlockBorderWrapper"] button[kind="primaryFormSubmit"]{
            background:linear-gradient(180deg, #37EAD0, #1FB39C) !important;
            color:#052420 !important;
            font-weight:700 !important;
            border:none !important;
            border-radius:8px !important;
            box-shadow:0 8px 24px -8px rgba(47,224,196,0.5);
        }

        /* expander ("Forgot password?") */
        div[data-testid="stVerticalBlockBorderWrapper"] details{
            border:1px solid var(--fm-line) !important;
            border-radius:8px !important;
            background:rgba(255,255,255,0.02) !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] summary{
            color:var(--fm-ink-dim) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_auth_pitch():
    """Left-hand marketing copy shown next to the sign-in card."""
    st.markdown(
        """
        <div style="display:flex; flex-direction:column;
                    justify-content:center; min-height:78vh;">
        <div id="fm-eyebrow">&#128269; FAILURE MINING</div>
        <div id="fm-title">AI-Powered Field Failure Analysis</div>
        <div id="fm-tagline">Field Returns Failure Mode Mining</div>
        <div id="fm-sub">Turn messy service-return comments into actionable
        failure modes and component insights.</div>
        <div id="fm-points">
            <div>&#128737;&#65039; PII automatically redacted before analysis</div>
            <div>&#9201;&#65039; Themes and trends in minutes, not days</div>
            <div>&#128190; Every analysis saved to your history</div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    # The OAuth callback is handled before these defaults are initialized.
    # On the first run after Google redirects back, handle_auth_redirect()
    # has already selected the destination page; do not overwrite it with
    # "auth" or the callback will appear to send the user back to sign-up.
    if _redirect_result == "login":
        st.session_state.page = "upload"
    elif _redirect_result == "recovery":
        st.session_state.page = "reset_password"
    else:
        st.session_state.page = "auth"
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


def do_logout():
    auth.logout_user()
    for key in (
        "user",
        "access_token",
        "refresh_token",
        "df",
        "cluster_labels",
        "k_used",
        "trend",
        "analysis_done",
        "source_name",
        "detection_info",
        "chat_history",
    ):
        st.session_state.pop(key, None)
    st.session_state.page = "auth"
    st.rerun()


# ---------------------------------------------------------------------------
# SLIDE 0 - LOGIN / SIGN UP
# ---------------------------------------------------------------------------
def render_auth_page():
    inject_auth_theme()

    pitch_col, card_col = st.columns([1.1, 1], gap="large")

    with pitch_col:
        render_auth_pitch()

    with card_col:
        with st.container(border=True):
            st.markdown("#### Welcome back")
            st.caption(
                "Sign in to continue -- a profile is created automatically "
                "the first time."
            )

            if not auth.is_configured():
                st.error(
                    "Supabase Auth isn't configured yet. Add `SUPABASE_URL` "
                    "and `SUPABASE_ANON_KEY` to `.streamlit/secrets.toml` "
                    "-- see README.md -> 'Setting up Supabase Auth'."
                )
                return

            # --- Continue with Google ---------------------------------
            _google_error = st.session_state.pop("google_auth_error", None)
            if _google_error:
                st.error(
                    f"Google sign-in didn't complete: {_google_error}\n\n"
                    "This is almost always a redirect-URL mismatch, not a "
                    "bug in this app -- double check: (1) `APP_URL` in "
                    "secrets matches a URL listed under Supabase -> "
                    "Authentication -> URL Configuration -> Redirect URLs, "
                    "and (2) in Google Cloud Console, the OAuth client's "
                    "Authorized redirect URI is Supabase's own callback "
                    "(`https://<project-ref>.supabase.co/auth/v1/callback`) "
                    "-- not this app's URL."
                )

            if _google_error:
                st.session_state.pop("google_login_url", None)
                st.session_state.pop("pkce_code_verifier", None)

            try:
                if "google_login_url" not in st.session_state:
                    st.session_state.google_login_url = auth.get_google_login_url()
                google_url = st.session_state.google_login_url

                # Use one plain anchor without target="_blank" so OAuth
                # starts in the current browser tab. Avoid st.link_button(),
                # which opens a new tab.
                st.markdown(
                    f'<a href="{google_url}" style="display:block;'
                    f"text-align:center;background:linear-gradient(180deg,"
                    f"#37EAD0,#1FB39C);color:#052420;padding:0.6rem 1rem;"
                    f"border-radius:0.5rem;text-decoration:none;"
                    f'font-weight:700;">🔵 Continue with Google</a>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Google sign-in isn't set up yet: {e}")

            st.divider()

            tab_login, tab_signup = st.tabs(["🔑 Log In", "🆕 Sign Up"])

            with tab_login:
                with st.form("login_form"):
                    email = st.text_input("Email", key="login_email")
                    password = st.text_input(
                        "Password",
                        type="password",
                        key="login_password",
                    )
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
                        reset_email = st.text_input(
                            "Enter your account email",
                            key="reset_email",
                        )
                        reset_submitted = st.form_submit_button("Send reset link")

                    if reset_submitted:
                        ok, msg = auth.request_password_reset(reset_email)
                        (st.success if ok else st.error)(msg)

            with tab_signup:
                st.caption(
                    "Creates your account with Supabase -- use a real "
                    "email, you'll need it to confirm/recover."
                )

                with st.form("signup_form"):
                    name = st.text_input("Full name", key="signup_name")
                    email = st.text_input("Email", key="signup_email")
                    password = st.text_input(
                        "Password (min 6 characters)",
                        type="password",
                        key="signup_password",
                    )
                    submitted = st.form_submit_button(
                        "Create Account",
                        type="primary",
                    )

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
        new_password = st.text_input(
            "New password (min 6 characters)",
            type="password",
            key="new_password",
        )
        confirm_password = st.text_input(
            "Confirm new password",
            type="password",
            key="confirm_password",
        )
        submitted = st.form_submit_button(
            "Update Password",
            type="primary",
        )

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
        "Signed in via "
        + ("Google" if user["provider"] == "google" else "Email & Password")
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

    uploaded_files = st.file_uploader(
        "📂 Upload your service notes files",
        type=[
            "csv",
            "xlsx",
            "xls",
            "tsv",
            "json",
            "pdf",
            "docx",
            "png",
            "jpg",
            "jpeg",
        ],
        accept_multiple_files=True,
        help=(
            "You can select multiple CSV, Excel, TSV, JSON, PDF, Word, "
            "or image files. All files are combined into one dataset and "
            "analyzed together for consistent failure themes."
        ),
    )

    st.caption(
        "Supported input: ✓ Multiple structured Failure Mining datasets  "
        "✓ Multiple raw customer/review files (CSV/XLSX, e.g. Google "
        "Reviews exports) ✓ PDF/Word/image files — all uploaded files "
        "are analyzed together."
    )

    auto_k = st.sidebar.checkbox(
        "Auto-select number of themes",
        value=True,
    )

    n_clusters = None

    if not auto_k:
        n_clusters = st.sidebar.slider(
            "Number of themes (clusters)",
            3,
            12,
            8,
        )

    if not uploaded_files:
        st.info(
            "Please upload one or more files "
            "(CSV, Excel, PDF, Word, or image)."
        )
        return

    st.success(f"{len(uploaded_files)} file(s) uploaded successfully.")

    for uploaded_file in uploaded_files:
        st.caption(f"• {uploaded_file.name}")

    analyze = st.button(
        "🔍 Analyze All Files",
        type="primary",
    )

    if not analyze:
        return

    import traceback

    try:
        # Read each upload once, then send the complete batch to the pipeline.
        # This is the key performance improvement: TF-IDF and KMeans run ONCE
        # over the combined dataset instead of once for every file.
        file_items = tuple(
            (
                uploaded_file.name,
                uploaded_file.getvalue(),
            )
            for uploaded_file in uploaded_files
        )

        with st.spinner(
            f"Analyzing {len(uploaded_files)} file(s): "
            "loading → sanitizing → vectorizing → clustering → labeling..."
        ):
            (
                df,
                cluster_labels,
                k_used,
                trend,
                detection_info,
            ) = load_and_process_batch(file_items, n_clusters)

        if df.empty:
            st.error("No files could be analyzed.")
            return

        source_name = ", ".join(
            name for name, _ in file_items
        )

        st.session_state.df = df
        st.session_state.cluster_labels = cluster_labels
        st.session_state.k_used = k_used
        st.session_state.trend = trend
        st.session_state.source_name = source_name
        st.session_state.detection_info = detection_info
        st.session_state.analysis_done = True

        # Keep the existing report + history feature.
        top_theme = (
            df["theme"].value_counts().idxmax()
            if len(df)
            else "N/A"
        )

        insights_full = build_insights(
            df,
            top_n=k_used,
        )

        with st.spinner("Preparing your analysis report..."):
            report_bytes = build_word_report(
                df,
                trend,
                insights_full,
                k_used,
                source_name=source_name,
            )

        safe_email = (
            st.session_state.user["email"]
            .replace("@", "_at_")
            .replace(".", "_")
        )

        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(
            REPORTS_DIR,
            f"{safe_email}_{timestamp}.docx",
        )

        with open(report_path, "wb") as f:
            f.write(report_bytes)

        auth.add_history(
            user_email=st.session_state.user["email"],
            source_name=source_name,
            total_notes=len(df),
            k_used=k_used,
            top_theme=top_theme,
            report_path=report_path,
        )

        go("hub")

    except ValueError as e:
        st.error(str(e))

    except Exception as e:
        st.error(f"Couldn't process the uploaded files: {e}")

        with st.expander("Full error details"):
            st.code(traceback.format_exc())


# ---------------------------------------------------------------------------
# Shared helper: apply sidebar filters, used by the hub + every section page
# ---------------------------------------------------------------------------
def get_filtered_data():
    df = st.session_state.df
    trend = st.session_state.trend

    models = sorted(df["product_model"].unique())

    selected_models = st.sidebar.multiselect(
        "Filter by product model",
        models,
        default=models,
    )

    date_min = df["date"].min()
    date_max = df["date"].max()

    date_range = st.sidebar.date_input(
        "Filter by date range",
        value=(date_min.date(), date_max.date()),
    )

    filtered = df[
        df["product_model"].isin(selected_models)
    ]

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        filtered = filtered[
            (filtered["date"] >= start)
            & (filtered["date"] <= end)
        ]

    trend_f = trend[
        trend["product_model"].isin(selected_models)
    ]

    trend_f = (
        trend_f.groupby(["month", "theme"])["count"]
        .sum()
        .reset_index()
    )

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
        f"Source: {st.session_state.source_name}  ·  "
        f"{len(df)} notes  ·  {k_used} failure themes discovered. "
        "Click a section below to open it."
    )

    detection_info = st.session_state.get(
        "detection_info",
        {"input_type": "structured"},
    )

    if detection_info.get("input_type") == "raw_reviews":
        ai_note = (
            "AI-enhanced extraction"
            if detection_info.get("ai_extraction_used")
            else "free keyword-based extraction "
            "(no AI API configured)"
        )

        st.info(
            f"🔎 Detected input type: **Raw Review Data**  ·  "
            f"Review column detected: "
            f"**{detection_info.get('review_column')}**  ·  "
            f"{detection_info.get('rows_dropped', 0)} "
            "empty/duplicate rows skipped  ·  "
            f"Component/severity extracted via {ai_note}."
        )

    inject_icon_css()

    for sec in SECTIONS:
        with st.container(border=True):
            c1, c2 = st.columns([1, 6])

            with c1:
                st.markdown(
                    f'<div class="icon-anim">{sec["icon"]}</div>',
                    unsafe_allow_html=True,
                )

            with c2:
                st.write("")

                if st.button(
                    sec["label"],
                    key=f"nav_{sec['key']}",
                    use_container_width=True,
                ):
                    go(sec["key"])

    if st.button("⬅️ Analyze different files"):
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
        "Service/repair notes → clustered themes → trends → actionable "
        "insights. All data below is synthetic (no real customer "
        "information)."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Notes (filtered)",
        len(filtered),
    )

    col2.metric(
        "Themes Discovered",
        k_used,
    )

    top_theme = (
        filtered["theme"].value_counts().idxmax()
        if len(filtered)
        else "N/A"
    )

    col3.metric(
        "Top Failure Mode",
        top_theme,
    )

    col4.metric(
        "Product Models",
        filtered["product_model"].nunique(),
    )

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
        height=480,
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# SECTION 3 - Trend Over Time by Theme
# ---------------------------------------------------------------------------
def render_trend():
    _, trend_f, _ = get_filtered_data()

    back_button()

    st.title("📈 Trend Over Time by Theme")

    fig_line = px.line(
        trend_f,
        x="month",
        y="count",
        color="theme",
        markers=True,
    )

    fig_line.update_layout(
        height=480,
        legend=dict(
            orientation="h",
            y=-0.3,
        ),
    )

    st.plotly_chart(
        fig_line,
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# SECTION 4 - Failure Counts by Product Model
# ---------------------------------------------------------------------------
def render_model_counts():
    filtered, _, _ = get_filtered_data()

    back_button()

    st.title("🏭 Failure Counts by Product Model")

    model_theme = (
        filtered.groupby(["product_model", "theme"])
        .size()
        .reset_index(name="count")
    )

    fig_stack = px.bar(
        model_theme,
        x="product_model",
        y="count",
        color="theme",
        barmode="stack",
    )

    fig_stack.update_layout(height=450)

    st.plotly_chart(
        fig_stack,
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# SECTION 5 - Actionable Insights Summary
# ---------------------------------------------------------------------------
def render_insights():
    filtered, _, _ = get_filtered_data()
    k_used = st.session_state.k_used

    back_button()

    st.title("💡 Actionable Insights Summary")

    insights = build_insights(
        filtered,
        top_n=k_used,
    )

    # --- Compact executive summary (Change 5) ---
    # insights is already sorted by frequency (most common theme first),
    # same ordering build_insights has always used -- so insights[0] is
    # the top failure theme without needing any new computation.
    top_insight = insights[0] if insights else None

    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
    sum_col1.metric("Reviews Analyzed", len(filtered))
    sum_col2.metric("Failure Themes", k_used)
    sum_col3.metric("Top Failure", top_insight["theme"] if top_insight else "N/A")
    sum_col4.metric(
        "Most Affected Model",
        top_insight["most_affected_model"] if top_insight else "N/A",
    )
    st.divider()

    for ins in insights:
        header_parts = [
            f"**{ins['theme']}** — "
            f"{ins['count']} notes "
            f"({ins['pct_of_total']}% of total) · "
            f"most affected: {ins['most_affected_model']}"
        ]

        # Change 3: only show serial range when it's actually meaningful.
        serial_range = ins.get("most_common_serial_range")
        if serial_range and serial_range != "Unknown":
            header_parts.append(f"serial range: {serial_range}")

        header = " · ".join(header_parts)

        with st.expander(header):
            if ins.get("tag_agreement_pct", 0) > 0:
                st.caption(
                    f"🏷️ Structured-tag agreement: "
                    f"{ins['tag_agreement_pct']}% "
                    "(how often the technician's own tag matched "
                    "this theme)"
                )

            # Change 4: "Why this pattern?" -- honest, TF-IDF/KMeans-only
            # explanation. No confidence scores, no semantic claims.
            if ins.get("key_terms"):
                st.markdown("**Why this pattern?**")
                st.caption("Key terms: " + " · ".join(ins["key_terms"]))
                st.caption(
                    f"{ins['count']} reviews were grouped into this "
                    "failure pattern."
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
        "note_id",
        "product_model",
        "serial_range",
        "date",
        "theme",
        "tag",
        "symptom_text_clean",
        "fix_text_clean",
    ]

    # component/severity only exist when the input was raw review data
    # (see review_preprocessor.py) -- shown only when present, so
    # structured datasets (which never have these columns) look exactly
    # as before.
    for extra_col in ("component", "severity"):
        if extra_col in filtered.columns:
            display_cols.append(extra_col)

    st.dataframe(
        filtered[display_cols],
        use_container_width=True,
        height=320,
    )

    st.divider()

    st.subheader("⬇️ Download Full Report")

    st.caption(
        "One Word document with the summary, all charts, and the "
        "actionable insights for the currently filtered view -- ready "
        "to share or submit."
    )

    insights = build_insights(
        filtered,
        top_n=k_used,
    )

    with st.spinner("Building your report..."):
        report_bytes = build_word_report(
            filtered,
            trend[
                trend["product_model"].isin(selected_models)
            ],
            insights,
            k_used,
            source_name=st.session_state.get(
                "source_name",
                "uploaded file",
            ),
        )

    st.download_button(
        label="📄 Download Analysis Report (Word)",
        data=report_bytes,
        file_name="failure_mode_analysis_report.docx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
        type="primary",
    )

    st.caption(
        "A snapshot of the full (unfiltered) report was also saved to "
        "your profile's History when you first ran this analysis."
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
        "Ask questions about the current analysis results (themes, "
        "trends, affected models). Answers are grounded only in this "
        "analysis -- powered by Groq."
    )

    if not chatbot.is_configured():
        st.warning(
            "The AI Assistant isn't configured yet -- set "
            "`GROQ_API_KEY` (as an environment variable or in "
            "`.streamlit/secrets.toml`). See README.md -> "
            "'Setting up the AI Assistant'."
        )
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    insights = build_insights(
        filtered,
        top_n=k_used,
    )

    context = chatbot.build_context(
        filtered,
        insights,
        trend_f,
        k_used,
        source_name=st.session_state.get(
            "source_name",
            "uploaded file",
        ),
    )

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    question = st.chat_input(
        "Ask about the failure themes, trends, or affected models..."
    )

    if question:
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = chatbot.ask_assistant(
                        context,
                        st.session_state.chat_history[:-1],
                        question,
                    )
                except RuntimeError as e:
                    answer = f"⚠️ {e}"

            st.markdown(answer)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

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
    st.caption(
        f"Past analyses run by {user['name']} ({user['email']})."
    )

    rows = auth.get_history(user["email"])

    if not rows:
        st.info(
            "No analyses yet -- upload a file to run your first one."
        )
        return

    for row in rows:
        when = (
            row["created_at"]
            .split(".")[0]
            .replace("T", " ")
        )

        with st.container(border=True):
            c1, c2 = st.columns([4, 1])

            with c1:
                st.markdown(
                    f"**{row['source_name']}** · {when}"
                )

                st.caption(
                    f"{row['total_notes']} notes · "
                    f"{row['k_used']} themes · "
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
                            mime=(
                                "application/vnd.openxmlformats-officedocument."
                                "wordprocessingml.document"
                            ),
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