"""
auth.py
-------
Handles user accounts via Supabase Auth:
- Email/password sign-in
- Continue with Google
- Forgot password
- Per-user analysis history stored in SQLite
"""

import os
import sqlite3
import datetime as dt
from contextlib import contextmanager

import streamlit as st
from supabase import create_client, Client

try:
    from supabase import ClientOptions
except ImportError:
    from supabase.lib.client_options import ClientOptions


DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "app.db",
)

# Keeps the PKCE verifier available when Google redirects into a new
# Streamlit session.
_LAST_PKCE_VERIFIER = None


# ---------------------------------------------------------------------------
# SUPABASE CLIENT
# ---------------------------------------------------------------------------
def _get_secret(key: str, default=None):
    """Read from Streamlit secrets first, then environment variables."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return os.environ.get(key, default)


@st.cache_resource
def get_client() -> Client:
    url = _get_secret("SUPABASE_URL")
    key = _get_secret("SUPABASE_ANON_KEY")

    if not url or not key:
        raise RuntimeError(
            "Supabase is not configured. Add SUPABASE_URL and "
            "SUPABASE_ANON_KEY to .streamlit/secrets.toml."
        )

    return create_client(
        url,
        key,
        options=ClientOptions(flow_type="pkce"),
    )


def is_configured() -> bool:
    try:
        get_client()
        return True
    except Exception:
        return False


def _redirect_url() -> str:
    """
    URL where Supabase sends the browser after Google sign-in or password
    recovery.
    """

    # First use an explicitly configured APP_URL.
    app_url = _get_secret("APP_URL")

    if app_url:
        return app_url

    # Use the deployed Streamlit Cloud URL when running there.
    if os.environ.get("STREAMLIT_SERVER_HEADLESS") == "true":
        return "https://fixminingai.streamlit.app/"

    # Local development fallback.
    return "http://localhost:8501"


# ---------------------------------------------------------------------------
# EMAIL AND PASSWORD AUTHENTICATION
# ---------------------------------------------------------------------------
def signup_user(email: str, password: str, name: str):
    """Create a new account through Supabase Auth."""

    email = (email or "").strip().lower()
    password = password or ""
    name = (name or "").strip()

    if not email or "@" not in email:
        return False, "Please enter a valid email address."

    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    if not name:
        return False, "Please enter your name."

    client = get_client()

    try:
        result = client.auth.sign_up(
            {
                "email": email,
                "password": password,
                "options": {
                    "data": {
                        "name": name,
                    },
                    "email_redirect_to": _redirect_url(),
                },
            }
        )
    except Exception as e:
        return False, _friendly_error(e)

    if result.user is None:
        return False, "Could not create account. Please try again."

    if result.session is None:
        return (
            True,
            "Account created! Check your inbox to confirm your email, "
            "then log in.",
        )

    st.session_state.user = _session_to_user_dict(result)
    _store_tokens(result)

    return True, "Account created! You're now signed in."


def login_user(email: str, password: str):
    """Sign in using email and password."""

    email = (email or "").strip().lower()
    password = password or ""

    client = get_client()

    try:
        result = client.auth.sign_in_with_password(
            {
                "email": email,
                "password": password,
            }
        )
    except Exception as e:
        return None, _friendly_error(e)

    _store_tokens(result)

    return _session_to_user_dict(result), "Welcome back!"


def request_password_reset(email: str):
    """Send a password-reset email."""

    email = (email or "").strip().lower()

    if not email or "@" not in email:
        return False, "Please enter a valid email address."

    client = get_client()

    try:
        client.auth.reset_password_for_email(
            email,
            {
                "redirect_to": _redirect_url(),
            },
        )
    except Exception as e:
        return False, _friendly_error(e)

    return (
        True,
        "If an account exists for that email, a reset link is on its way.",
    )


def update_password(new_password: str):
    """Update the password after a password-recovery redirect."""

    new_password = new_password or ""

    if len(new_password) < 6:
        return False, "Password must be at least 6 characters."

    client = get_client()

    try:
        client.auth.update_user(
            {
                "password": new_password,
            }
        )
    except Exception as e:
        return False, _friendly_error(e)

    return True, "Password updated! You can now log in with your new password."


# ---------------------------------------------------------------------------
# GOOGLE OAUTH
# ---------------------------------------------------------------------------
def _pkce_storage_key(client: Client) -> str:
    return f"{client.auth._storage_key}-code-verifier"


def get_google_login_url() -> str:
    """
    Generate the Google OAuth URL and save the PKCE verifier.
    """

    global _LAST_PKCE_VERIFIER

    client = get_client()

    result = client.auth.sign_in_with_oauth(
        {
            "provider": "google",
            "options": {
                "redirect_to": _redirect_url(),
                "query_params": {
                    "prompt": "select_account",
                },
            },
        }
    )

    verifier = client.auth._storage.get_item(
        _pkce_storage_key(client)
    )

    _LAST_PKCE_VERIFIER = verifier
    st.session_state["pkce_code_verifier"] = verifier

    return result.url


def handle_auth_redirect():
    """
    Process the authorization code returned by Supabase after Google login.

    Supabase Python expects exchange_code_for_session() to receive a
    dictionary containing the auth code:

        {"auth_code": code}
    """

    params = st.query_params
    code = params.get("code")

    if not code:
        return None

    is_recovery = params.get("type") == "recovery"
    client = get_client()

    try:
        verifier = (
            st.session_state.get("pkce_code_verifier")
            or _LAST_PKCE_VERIFIER
        )

        if verifier:
            # Restore the verifier before exchanging the authorization code.
            client.auth._storage.set_item(
                _pkce_storage_key(client),
                verifier,
            )

        # Important: supabase-py expects a dictionary here.
        result = client.auth.exchange_code_for_session(
            {
                "auth_code": code,
            }
        )

    except Exception as e:
        st.session_state["google_auth_error"] = str(e)
        st.query_params.clear()
        return None

    # Remove the one-time authorization code from the browser URL.
    st.query_params.clear()

    st.session_state.pop("pkce_code_verifier", None)
    st.session_state.pop("google_login_url", None)

    _store_tokens(result)

    if is_recovery:
        return "recovery"

    st.session_state.user = _session_to_user_dict(result)

    return "login"


def logout_user():
    client = get_client()

    try:
        client.auth.sign_out()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _store_tokens(result):
    session = getattr(result, "session", None)

    if session:
        st.session_state.access_token = session.access_token
        st.session_state.refresh_token = session.refresh_token


def _session_to_user_dict(result):
    user = result.user
    metadata = user.user_metadata or {}

    email = user.email or ""
    name = (
        metadata.get("name")
        or metadata.get("full_name")
        or email.split("@")[0]
    )

    provider = "password"

    if any(
        identity.provider == "google"
        for identity in (user.identities or [])
    ):
        provider = "google"

    return {
        "id": user.id,
        "email": email,
        "name": name,
        "provider": provider,
    }


def _friendly_error(error: Exception) -> str:
    message = str(error)

    if "Invalid login credentials" in message:
        return "Incorrect email or password."

    if "User already registered" in message:
        return (
            "An account with this email already exists. "
            "Please log in instead."
        )

    if "Email not confirmed" in message:
        return (
            "Please confirm your email first -- check your inbox "
            "for the confirmation link."
        )

    return message


# ---------------------------------------------------------------------------
# SQLITE ANALYSIS HISTORY
# ---------------------------------------------------------------------------
def _ensure_data_dir():
    os.makedirs(
        os.path.dirname(DB_PATH),
        exist_ok=True,
    )


@contextmanager
def get_conn():
    _ensure_data_dir()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create the analysis history table if it does not exist."""

    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                source_name TEXT NOT NULL,
                total_notes INTEGER NOT NULL,
                k_used INTEGER NOT NULL,
                top_theme TEXT,
                report_path TEXT,
                created_at TEXT NOT NULL
            )
            """
        )


def add_history(
    user_email: str,
    source_name: str,
    total_notes: int,
    k_used: int,
    top_theme: str,
    report_path: str = None,
):
    """Save one completed analysis to the local history database."""

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO history (
                user_email,
                source_name,
                total_notes,
                k_used,
                top_theme,
                report_path,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_email.lower(),
                source_name,
                total_notes,
                k_used,
                top_theme,
                report_path,
                dt.datetime.now().isoformat(),
            ),
        )


def get_history(user_email: str):
    """Return analysis history for one user."""

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM history
            WHERE user_email = ?
            ORDER BY created_at DESC
            """,
            (user_email.lower(),),
        ).fetchall()

    return [dict(row) for row in rows]