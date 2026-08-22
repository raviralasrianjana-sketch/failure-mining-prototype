"""
auth.py
-------
Handles user accounts via Supabase Auth (real email/password accounts,
"Continue with Google", and "Forgot password"), plus per-user analysis
history (still stored locally in SQLite, keyed by the user's email).

WHY SUPABASE?
The old version of this file hand-rolled email/password accounts with
bcrypt in a local SQLite file. That worked, but it had two gaps:
  - No way to recover a forgotten password.
  - No real "Continue with Google" (so people used throwaway emails).
Supabase Auth is a free, hosted auth service that gives us both of
these for free, plus secure password storage (we never touch raw
passwords -- Supabase does).

ONE-TIME SETUP required from whoever runs this app -- see
README.md -> "Setting up Supabase Auth". Until that's done, the auth
page will show a clear setup message instead of crashing.
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

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "app.db")

# PKCE verifier must survive Google opening in a new tab (new Streamlit
# session). session_state does not carry over; this process-level store does.
_LAST_PKCE_VERIFIER = None


# ---------------------------------------------------------------------------
# SUPABASE CLIENT
# ---------------------------------------------------------------------------
def _get_secret(key: str, default=None):
    """Reads from .streamlit/secrets.toml if present, else from a real
    environment variable (useful when deploying outside Streamlit Cloud)."""
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
            "Supabase is not configured. Add SUPABASE_URL and SUPABASE_ANON_KEY "
            "to .streamlit/secrets.toml (see README.md -> 'Setting up Supabase Auth')."
        )
    # flow_type="pkce" is required for server-side apps like this one --
    # without it, Google's redirect can come back with tokens in the URL
    # fragment (#access_token=...) instead of a ?code=... query param,
    # and Streamlit's server-side code never sees a fragment at all.
    return create_client(url, key, options=ClientOptions(flow_type="pkce"))


def is_configured() -> bool:
    try:
        get_client()
        return True
    except Exception:
        return False


def _redirect_url() -> str:
    """
    Where Supabase/Google should send the browser back to after Google
    sign-in, or after the user clicks the link in a 'reset password'
    email. This MUST also be added in the Supabase dashboard under
    Authentication -> URL Configuration -> Redirect URLs, and must match
    wherever this app is actually hosted (localhost while developing,
    your real deployed URL in production).
    """
    return _get_secret("APP_URL", "http://localhost:8501")


# ---------------------------------------------------------------------------
# EMAIL + PASSWORD
# ---------------------------------------------------------------------------
def signup_user(email: str, password: str, name: str):
    """Creates a new account via Supabase. Returns (success, message)."""
    email = (email or "").strip().lower()
    name = (name or "").strip()
    if not email or "@" not in email:
        return False, "Please enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if not name:
        return False, "Please enter your name."

    client = get_client()
    try:
        result = client.auth.sign_up({
            "email": email,
            "password": password,
            "options": {"data": {"name": name}, "email_redirect_to": _redirect_url()},
        })
    except Exception as e:
        return False, _friendly_error(e)

    if result.user is None:
        return False, "Could not create account. Please try again."
    if result.session is None:
        # "Confirm email" is on in Supabase -- user must click the link in
        # their inbox before they can log in for the first time.
        return True, "Account created! Check your inbox to confirm your email, then log in."

    st.session_state.user = _session_to_user_dict(result)
    _store_tokens(result)
    return True, "Account created! You're now signed in."


def login_user(email: str, password: str):
    """Verifies an email/password login. Returns (user_dict_or_None, message)."""
    email = (email or "").strip().lower()
    client = get_client()
    try:
        result = client.auth.sign_in_with_password({"email": email, "password": password})
    except Exception as e:
        return None, _friendly_error(e)

    _store_tokens(result)
    return _session_to_user_dict(result), "Welcome back!"


def request_password_reset(email: str):
    """Sends a 'forgot password' email via Supabase. Returns (success, message)."""
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        return False, "Please enter a valid email address."

    client = get_client()
    try:
        client.auth.reset_password_for_email(email, {"redirect_to": _redirect_url()})
    except Exception as e:
        return False, _friendly_error(e)
    # Supabase reports success here even for emails with no account, so we
    # don't leak which addresses are registered -- that's expected behavior.
    return True, "If an account exists for that email, a reset link is on its way."


def update_password(new_password: str):
    """Sets a new password. Only works while the user has an active session
    established from a password-reset email link (see handle_auth_redirect)."""
    if len(new_password) < 6:
        return False, "Password must be at least 6 characters."
    client = get_client()
    try:
        client.auth.update_user({"password": new_password})
    except Exception as e:
        return False, _friendly_error(e)
    return True, "Password updated! You can now log in with your new password."


# ---------------------------------------------------------------------------
# GOOGLE OAUTH ("Continue with Google")
# ---------------------------------------------------------------------------
def _pkce_storage_key(client: Client) -> str:
    return f"{client.auth._storage_key}-code-verifier"


def get_google_login_url() -> str:
    """Returns the URL to send the browser to in order to sign in with Google.

    The PKCE verifier is saved both in session_state and in a process-level
    variable so the code exchange still works after Google sends the browser
    back (including when Streamlit opens Google in a new tab).
    """
    global _LAST_PKCE_VERIFIER
    client = get_client()
    result = client.auth.sign_in_with_oauth({
        "provider": "google",
        "options": {
            "redirect_to": _redirect_url(),
            "query_params": {"prompt": "select_account"},
        },
    })
    verifier = client.auth._storage.get_item(_pkce_storage_key(client))
    _LAST_PKCE_VERIFIER = verifier
    st.session_state["pkce_code_verifier"] = verifier
    return result.url


def handle_auth_redirect():
    """
    Call this once, near the very top of app.py, on every script run.
    After Google sign-in or clicking a password-reset email link,
    Supabase sends the browser back to this app with a `?code=...` (and
    sometimes `&type=recovery`) query param. This exchanges that code for
    a real session.

    Returns:
      "login"    - a normal session was established (e.g. Google sign-in)
      "recovery" - this is a password-reset link; caller should show the
                   "choose a new password" screen instead of logging in
      None       - nothing to handle on this run
    """
    params = st.query_params
    code = params.get("code")
    if not code:
        return None

    is_recovery = params.get("type") == "recovery"
    client = get_client()
    try:
        exchange = {"auth_code": code, "redirect_to": _redirect_url()}
        verifier = st.session_state.get("pkce_code_verifier") or _LAST_PKCE_VERIFIER
        if verifier:
            exchange["code_verifier"] = verifier
        result = client.auth.exchange_code_for_session(exchange)
    except Exception as e:
        # Don't swallow this -- a failed exchange used to just silently
        # bounce back to the login page with zero explanation, which is
        # indistinguishable from "Continue with Google" doing nothing at
        # all. Stash the real error so render_auth_page() can show it.
        # The two most common causes:
        #   1. APP_URL (secrets) doesn't exactly match a URL registered
        #      under Supabase -> Authentication -> URL Configuration ->
        #      Redirect URLs.
        #   2. In Google Cloud Console, the OAuth client's "Authorized
        #      redirect URI" must point to Supabase's own callback
        #      (https://<project-ref>.supabase.co/auth/v1/callback), not
        #      to this app's URL -- this app's URL only belongs in
        #      Supabase's redirect-URL allowlist, not Google's.
        st.session_state["google_auth_error"] = str(e)
        st.query_params.clear()
        return None

    st.query_params.clear()  # keep the URL clean once the code is consumed
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
# helpers
# ---------------------------------------------------------------------------
def _store_tokens(result):
    if getattr(result, "session", None):
        st.session_state.access_token = result.session.access_token
        st.session_state.refresh_token = result.session.refresh_token


def _session_to_user_dict(result):
    u = result.user
    meta = u.user_metadata or {}
    name = meta.get("name") or meta.get("full_name") or u.email.split("@")[0]
    provider = "google" if any(i.provider == "google" for i in (u.identities or [])) else "password"
    return {"id": u.id, "email": u.email, "name": name, "provider": provider}


def _friendly_error(e: Exception) -> str:
    msg = str(e)
    if "Invalid login credentials" in msg:
        return "Incorrect email or password."
    if "User already registered" in msg:
        return "An account with this email already exists. Please log in instead."
    if "Email not confirmed" in msg:
        return "Please confirm your email first -- check your inbox for the confirmation link."
    return msg


# ---------------------------------------------------------------------------
# HISTORY (unchanged behavior -- still local SQLite, keyed by email)
# ---------------------------------------------------------------------------
def _ensure_data_dir():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


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
    """Creates the history table if it doesn't exist yet. Safe to call on every app start.
    (User accounts themselves now live in Supabase, not here.)"""
    with get_conn() as conn:
        conn.execute("""
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
        """)


def add_history(user_email: str, source_name: str, total_notes: int, k_used: int,
                 top_theme: str, report_path: str = None):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO history
               (user_email, source_name, total_notes, k_used, top_theme, report_path, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_email.lower(), source_name, total_notes, k_used, top_theme,
             report_path, dt.datetime.now().isoformat()),
        )


def get_history(user_email: str):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM history WHERE user_email = ? ORDER BY created_at DESC",
            (user_email.lower(),),
        ).fetchall()
    return [dict(r) for r in rows]