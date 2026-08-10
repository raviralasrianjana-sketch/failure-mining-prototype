"""
auth.py
-------
Handles user accounts (email/password signup+login), Google sign-in, and
per-user analysis history.

WHY SQLITE?
This is a hackathon MVP -- we don't want to stand up a real database
server. SQLite stores everything in one local file (data/app.db), zero
setup required, and is good enough to demo "real" accounts + history.
For production, swap this for Postgres/MySQL; the functions below are
the only place that would need to change (the rest of the app just
calls these functions).

WHY BCRYPT?
Passwords must NEVER be stored as plain text. bcrypt is a purpose-built,
battle-tested password hashing algorithm (deliberately slow, so a stolen
database of hashes is hard to brute-force) -- the standard choice here.

GOOGLE SIGN-IN
"Continue with Google" uses Streamlit's built-in st.login() / st.logout()
/ st.user, which implements OpenID Connect under the hood. This needs a
ONE-TIME setup step from whoever runs the app: creating an OAuth Client
ID in Google Cloud Console and adding it to .streamlit/secrets.toml (see
README.md -> "Setting up Google Sign-In"). If that isn't configured yet,
clicking the Google button shows a clear message instead of crashing --
email/password login works fully either way, with no setup needed.
"""

import os
import sqlite3
import datetime as dt
from contextlib import contextmanager

import bcrypt

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "app.db")


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
    """Creates the users/history tables if they don't exist yet. Safe to call on every app start."""
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT,                          -- NULL for Google-only accounts
                provider TEXT NOT NULL DEFAULT 'password',    -- 'password' or 'google'
                created_at TEXT NOT NULL
            )
        """)
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


# ---------------------------------------------------------------------------
# EMAIL + PASSWORD ACCOUNTS
# ---------------------------------------------------------------------------
def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def signup_user(email: str, password: str, name: str):
    """Creates a new email/password account + profile. Returns (success, message)."""
    email = (email or "").strip().lower()
    name = (name or "").strip()

    if not email or "@" not in email:
        return False, "Please enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if not name:
        return False, "Please enter your name."

    with get_conn() as conn:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            return False, "An account with this email already exists. Please log in instead."
        conn.execute(
            "INSERT INTO users (email, name, password_hash, provider, created_at) VALUES (?, ?, ?, ?, ?)",
            (email, name, _hash_password(password), "password", dt.datetime.now().isoformat()),
        )
    return True, "Account created! You're now signed in."


def login_user(email: str, password: str):
    """Verifies an email/password login. Returns (user_dict_or_None, message)."""
    email = (email or "").strip().lower()

    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    if row is None:
        return None, "No account found with that email. Please sign up first."
    if row["provider"] != "password":
        return None, "This email is registered via Google Sign-In. Use 'Continue with Google' instead."
    if not _verify_password(password, row["password_hash"]):
        return None, "Incorrect password. Please try again."

    return dict(row), "Welcome back!"


def get_or_create_google_profile(email: str, name: str):
    """
    Called right after a successful Google sign-in (st.user is populated).
    Makes sure a profile row exists for this Google account, so history
    and "My Profile" work the same way as for email/password users.
    """
    email = (email or "").strip().lower()
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO users (email, name, password_hash, provider, created_at) VALUES (?, ?, ?, ?, ?)",
                (email, name or email.split("@")[0], None, "google", dt.datetime.now().isoformat()),
            )
            row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    return dict(row)


# ---------------------------------------------------------------------------
# HISTORY (per-profile record of past analyses)
# ---------------------------------------------------------------------------
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
