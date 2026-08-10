"""
chatbot.py
----------
"Failure Mining AI Assistant" -- a chat interface grounded in the actual
analysis results produced by pipeline.py. This is NOT a general chatbot:
it only ever answers from a text summary of the real analysis, never
from general knowledge about failure modes.

WHY A SEPARATE FILE?
Same reasoning as auth.py -- keeps LLM/chat logic isolated from the UI
(app.py) and the ML pipeline (pipeline.py). Nothing about the existing
clustering/analysis has to change to add this feature, and this file can
be deleted with zero impact on the rest of the app.

HOW IT'S GROUNDED IN THE DATA (not a generic chatbot)
We never send the raw dataset to the LLM. build_context() turns the
already-computed pipeline outputs (theme counts/percentages, most
affected model + serial range per theme, tag agreement, a couple of
sanitized example notes per theme, and a short trend snapshot) into one
compact text block -- a few hundred words, not thousands of rows. That
block is the LLM's ONLY source of truth: the system prompt explicitly
tells it to answer only from that context and say so when something
isn't in it, instead of inventing numbers or component names.

LLM PROVIDER
Uses the Groq API (fast inference, generous no-credit-card free tier).
Config is read from (in order): OS environment variables, then
st.secrets (so it also works on Streamlit Cloud). Needed values:
  GROQ_API_KEY   - your Groq API key (from https://console.groq.com/keys)
  GROQ_MODEL     - optional, defaults to "llama-3.3-70b-versatile"
No config -> the assistant shows a friendly setup message instead of
crashing; the rest of the app (upload, clustering, dashboard, history)
is unaffected.
"""

import os

import streamlit as st

_DEFAULT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT_TEMPLATE = """You are the "Failure Mining AI Assistant" -- an AI analyst built \
specifically to explain the results of THIS field-returns failure mode analysis. You are not a \
general-purpose chatbot and must not use outside/general knowledge about products or failure types.

Rules you must always follow:
- Only use the ANALYSIS CONTEXT below to answer. Never invent theme names, numbers, percentages, \
component names, or root causes that are not present in the context.
- If the answer isn't in the context, say clearly that it isn't available in the current analysis \
results, instead of guessing.
- Keep answers concise (a few sentences, or a short list) and analyst-toned. Reference the actual \
theme names, counts/percentages, and models/serial ranges from the context.
- When asked "why" a theme was identified, or to "explain" it, base the explanation on that theme's \
example notes / keywords in the context, not general assumptions.
- When asked to compare themes, use the counts/percentages/models actually given for each.

ANALYSIS CONTEXT
-----------------
{context}
"""


def build_context(df, insights, trend, k_used, source_name) -> str:
    """
    Compact, text-only summary of the analysis results (NOT the raw
    dataset) -- this is what actually gets sent to the LLM.

    Args mirror what app.py already has in hand after run_pipeline():
      df       - the full processed dataframe (used only for aggregate
                 counts here, individual rows are never sent)
      insights - output of pipeline.build_insights(): list of dicts with
                 theme, count, pct_of_total, most_affected_model,
                 most_common_serial_range, tag_agreement_pct, examples
      trend    - output of pipeline.build_trend_table()
      k_used   - number of clusters/themes chosen
      source_name - uploaded filename, for context only
    """
    lines = [
        f"Source file: {source_name}",
        f"Total notes analyzed: {len(df)}",
        f"Number of failure themes discovered: {k_used}",
        "",
        "--- FAILURE THEMES (sorted by frequency, most common first) ---",
    ]

    for ins in insights:
        lines.append(
            f"- {ins['theme']}: {ins['count']} notes ({ins['pct_of_total']}% of total notes). "
            f"Most affected product model: {ins['most_affected_model']}. "
            f"Most common serial range: {ins.get('most_common_serial_range', 'Unknown')}. "
            f"Structured-tag agreement: {ins.get('tag_agreement_pct', 0)}%."
        )
        for ex in ins["examples"][:2]:
            lines.append(f'    example note: "{ex}"')

    if trend is not None and len(trend):
        lines.append("")
        lines.append("--- RECENT TREND (last 3 months on record, per theme) ---")
        trend_sorted = trend.sort_values("month")
        for theme, grp in trend_sorted.groupby("theme"):
            last3 = grp.tail(3)
            points = ", ".join(f"{row['month']}: {row['count']}" for _, row in last3.iterrows())
            lines.append(f"- {theme}: {points}")

    lines.append("")
    lines.append("--- TOP THEMES PER PRODUCT MODEL ---")
    model_theme = df.groupby(["product_model", "theme"]).size().reset_index(name="count")
    for model, grp in model_theme.groupby("product_model"):
        top = grp.sort_values("count", ascending=False).head(3)
        parts = ", ".join(f"{row['theme']} ({row['count']})" for _, row in top.iterrows())
        lines.append(f"- {model}: {parts}")

    return "\n".join(lines)


def _get_setting(name, default=None):
    """Checks env var first (local dev), then st.secrets (Streamlit Cloud)."""
    val = os.environ.get(name)
    if val:
        return val
    try:
        val = st.secrets.get(name)
    except Exception:
        val = None
    return val if val else default


def _groq_config():
    return {
        "api_key": _get_setting("GROQ_API_KEY"),
        "model": _get_setting("GROQ_MODEL", _DEFAULT_MODEL),
    }


def is_configured() -> bool:
    cfg = _groq_config()
    return bool(cfg["api_key"] and cfg["model"])


def ask_assistant(context: str, chat_history: list, question: str) -> str:
    """
    chat_history: list of {"role": "user"/"assistant", "content": str} --
    prior turns in THIS session, used so follow-ups like "why?" resolve
    against the previous answer.

    Returns the assistant's reply text. Raises RuntimeError with a
    user-friendly message on any failure (missing config, missing
    package, API error) -- callers should catch this and show it, not
    crash the app.
    """
    cfg = _groq_config()
    if not is_configured():
        raise RuntimeError(
            "The AI Assistant isn't configured yet -- set GROQ_API_KEY "
            "(as an environment variable or in .streamlit/secrets.toml). "
            "See README.md -> 'Setting up the AI Assistant'."
        )

    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError(
            "The 'groq' package isn't installed. Run `pip install -r requirements.txt`."
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(context=context)}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": question})

    try:
        client = Groq(api_key=cfg["api_key"])
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=messages,
            temperature=0.2,
            max_tokens=800,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"The AI Assistant couldn't reach Groq right now. ({e})")