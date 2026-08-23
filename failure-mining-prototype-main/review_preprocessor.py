"""
review_preprocessor.py
-----------------------
Adaptation layer that converts UNSTRUCTURED review data (e.g. a raw
Google Reviews export) into the same standardized shape the existing
Failure Mining pipeline (pipeline.py) already expects:

    product_model, date, symptom_text, fix_text
    (+ serial_range, tag -- pipeline.py's load_data() already defaults
    these to "Unknown" / "" when missing, so we don't need to invent
    them here)

WHEN THIS FILE RUNS
Only when pipeline.py's load_data() determines an uploaded CSV/Excel/
TSV/JSON file does NOT already have the columns the existing pipeline
needs (see is_structured_failure_data). Files that already have those
columns skip this file entirely and go through the original, unchanged
path -- so results for already-working datasets never change.

WHY A SEPARATE FILE INSTEAD OF ADDING THIS TO pipeline.py?
pipeline.py's own docstring says to keep the ML "brain" isolated. This
file is a pure *adapter*: it never touches clustering, vectorizing, or
labeling. It only reshapes raw review data into the table pipeline.py
already knows how to consume.

AI/LLM USE (OPTIONAL)
An LLM can be used to pull out a component/symptom/severity from each
review for nicer display. This is enrichment, not a requirement -- the
clustering step only ever needs `symptom_text`, so if no AI API key is
configured, we fall back to a small built-in keyword dictionary instead
of blocking raw-review analysis entirely. The API key is NEVER hardcoded
here -- see _get_ai_api_key().
"""

import os
import re
from datetime import datetime

import pandas as pd


# ---------------------------------------------------------------------------
# STRUCTURED-DATA DETECTION
# ---------------------------------------------------------------------------
REQUIRED_PIPELINE_COLUMNS = {"product_model", "date", "symptom_text"}


def is_structured_failure_data(df: pd.DataFrame) -> bool:
    """
    True if the uploaded file already has the exact columns the existing
    pipeline requires. This check is intentionally identical to the one
    pipeline.py's load_data() already performs -- so "already structured"
    data is guaranteed to be routed straight through, unchanged.
    """
    return REQUIRED_PIPELINE_COLUMNS.issubset(set(df.columns))


# ---------------------------------------------------------------------------
# COLUMN DETECTION (arbitrary column names -> the fields we need)
# ---------------------------------------------------------------------------
_REVIEW_COLUMN_HINTS = [
    "review_text", "review", "comment", "comments", "feedback",
    "description", "remarks", "notes", "message", "body", "text",
]

_METADATA_COLUMN_HINTS = {
    "product_model": ["product_model", "model", "product", "car_model", "vehicle"],
    "date": ["review_date", "date", "created_at", "timestamp"],
    "rating": ["rating", "stars", "score"],
    "location": ["location", "city", "store", "branch"],
    "customer": ["customer", "reviewer", "author", "user", "name"],
}


def detect_review_column(df: pd.DataFrame):
    """
    Finds the column most likely to contain the actual review/complaint
    text.
      1. Name matching -- common review-column names, case-insensitive.
         A name match is only accepted if that column's text is
         reasonably long on average (avoids matching e.g. a "notes" ID
         code column that happens to share a hint word).
      2. Content fallback -- if no name matches, use whichever text
         (object-dtype) column has the longest average string length,
         since real review text reads like sentences, not short labels.
    Returns None if nothing plausible is found.
    """
    lower_cols = {c.lower().strip(): c for c in df.columns}

    for hint in _REVIEW_COLUMN_HINTS:
        for lower_name, original_name in lower_cols.items():
            if hint == lower_name or hint in lower_name:
                sample = df[original_name].dropna().astype(str)
                if len(sample) and sample.str.len().mean() >= 8:
                    return original_name

    text_cols = df.select_dtypes(include="object").columns
    best_col, best_avg_len = None, 0
    for col in text_cols:
        sample = df[col].dropna().astype(str)
        if len(sample) == 0:
            continue
        avg_len = sample.str.len().mean()
        if avg_len > best_avg_len:
            best_col, best_avg_len = col, avg_len

    # Require a minimum average length so a short label column (e.g.
    # "city") isn't mistaken for a review column.
    if best_col is not None and best_avg_len >= 15:
        return best_col

    return None


def detect_metadata_columns(df: pd.DataFrame) -> dict:
    """
    Looks for optional useful columns (model, date, rating, location,
    customer) by name, so they can be carried through instead of
    discarded. Returns {standard_name: actual_column_name_or_None}.
    """
    lower_cols = {c.lower().strip(): c for c in df.columns}
    found = {}
    for standard_name, hints in _METADATA_COLUMN_HINTS.items():
        found[standard_name] = None
        for hint in hints:
            for lower_name, original_name in lower_cols.items():
                if hint == lower_name or hint in lower_name:
                    found[standard_name] = original_name
                    break
            if found[standard_name]:
                break
    return found


# ---------------------------------------------------------------------------
# TEXT CLEANING (light-touch, non-destructive)
# ---------------------------------------------------------------------------
def clean_review_text(text) -> str:
    """
    Normalizes whitespace only. Deliberately does NOT strip emojis,
    punctuation, or capitalization -- those can carry meaning (e.g. "!!!"
    or a frustrated emoji signals severity), and the brief explicitly
    says not to discard information that could help identify failures.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # collapse repeated whitespace/newlines
    return text


# ---------------------------------------------------------------------------
# OPTIONAL AI EXTRACTION (component / symptom / severity), with a free
# keyword-based fallback that needs no API key at all.
# ---------------------------------------------------------------------------
_KEYWORD_COMPONENTS = {
    "Engine": ["engine", "motor", "stall", "rpm"],
    "Brakes": ["brake", "braking"],
    "Battery": ["battery", "charge", "charging"],
    "Air Conditioning / Cooling": ["ac ", "a/c", "air condition", "cooling", "cooler"],
    "Display / Screen": ["screen", "display", "touchscreen"],
    "Noise / Vibration": ["noise", "rattle", "vibrat", "squeak"],
    "Electrical": ["electrical", "wiring", "short circuit", "fuse"],
    "Software": ["software", "app", "firmware", "update", "bug", "crash", "freeze"],
    "Body / Exterior": ["dent", "scratch", "paint", "rust", "housing", "casing"],
}


def _keyword_extract_component(text: str) -> str:
    lowered = text.lower()
    for component, keywords in _KEYWORD_COMPONENTS.items():
        if any(kw in lowered for kw in keywords):
            return component
    return "Unknown"


def _get_ai_api_key():
    """
    Reads an AI API key from (in order): the OPENAI_API_KEY environment
    variable, or st.secrets["openai"]["api_key"] if running inside
    Streamlit with a configured secrets.toml. Returns None if neither is
    set -- callers must treat that as "use the free keyword fallback."
    NEVER hardcode a key here.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("openai", {}).get("api_key")
    except Exception:
        return None


_openai_client_cache = {}


def _get_openai_client(api_key: str):
    """Reuse one OpenAI client per API key instead of creating a new one
    on every single row -- cheap win now that calls run concurrently."""
    if api_key not in _openai_client_cache:
        from openai import OpenAI
        _openai_client_cache[api_key] = OpenAI(api_key=api_key)
    return _openai_client_cache[api_key]


def llm_extract_fields(review_text: str):
    """
    Asks an LLM to identify the component, symptom, and severity in a
    single review, using ONLY what's in the text (never inventing a
    component that isn't mentioned). Returns None (never raises) if no
    API key is configured or the call fails for any reason -- callers
    should fall back to the keyword-based extraction instead.
    """
    api_key = _get_ai_api_key()
    if not api_key:
        return None

    try:
        client = _get_openai_client(api_key)

        prompt = (
            "You extract structured info from a single product/service review. "
            "Only use information present in the review -- if something isn't "
            "mentioned, answer \"Unknown\", never guess or invent it.\n\n"
            f"Review: \"{review_text}\"\n\n"
            "Respond with exactly these three lines, nothing else:\n"
            "Component: <short component/part name or Unknown>\n"
            "Symptom: <short failure symptom or Unknown>\n"
            "Severity: <Low, Medium, High, or Unknown>"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
        )
        text = response.choices[0].message.content or ""

        fields = {"component": "Unknown", "severity": "Unknown"}
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key, value = key.strip().lower(), value.strip()
            if key == "component" and value:
                fields["component"] = value
            elif key == "severity" and value:
                fields["severity"] = value
        return fields
    except Exception:
        return None  # never let an AI/network hiccup break the app


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------
def preprocess_raw_reviews(df: pd.DataFrame, source_filename: str = "reviews"):
    """
    Converts a raw/unstructured review DataFrame into the standardized
    shape pipeline.py expects: product_model, date, symptom_text,
    fix_text (serial_range/tag are left for pipeline.py's own defaults).

    Returns (standardized_df, detection_info). detection_info is a small
    dict describing what was detected/done -- app.py uses it to show a
    "Detected input type: Raw Review Data / Review column: X" message.
    """
    review_col = detect_review_column(df)
    if review_col is None:
        raise ValueError(
            "Couldn't find a review/comment/description column in this file. "
            "Please upload a file with a column containing review text "
            "(e.g. named 'review', 'comment', 'feedback', or 'description'), "
            "or a file already structured for Failure Mining "
            "(with product_model, date, symptom_text columns)."
        )

    meta_cols = detect_metadata_columns(df)

    working = df.copy()
    working["_clean_review"] = working[review_col].apply(clean_review_text)

    # Data quality: drop rows with no usable review text, and drop exact
    # duplicate reviews (keep the first occurrence) -- both explicitly
    # called out as things to handle. We do NOT drop short reviews --
    # "AC stopped cooling" is short but perfectly usable.
    rows_in = len(working)
    working = working[working["_clean_review"].str.len() > 0]
    working = working.drop_duplicates(subset="_clean_review", keep="first")
    rows_dropped = rows_in - len(working)

    ai_used = _get_ai_api_key() is not None
    review_texts = list(working["_clean_review"])
    components, severities = [], []
    if ai_used:
        # Run the per-row LLM calls concurrently instead of one blocking
        # network round-trip at a time -- this is the main speedup for
        # "Analyze" on raw review files, since hundreds/thousands of
        # sequential calls otherwise means minutes of dead time. These are
        # I/O-bound (waiting on the network, not the CPU), so a higher
        # worker count is safe and directly cuts wall-clock time -- capped
        # so a tiny upload doesn't spin up more threads than it has rows.
        from concurrent.futures import ThreadPoolExecutor

        max_workers = min(24, max(1, len(review_texts)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            all_fields = list(executor.map(llm_extract_fields, review_texts))
        for text, fields in zip(review_texts, all_fields):
            if fields:
                components.append(fields["component"])
                severities.append(fields["severity"])
            else:
                components.append(_keyword_extract_component(text))
                severities.append("Unknown")
    else:
        for text in review_texts:
            components.append(_keyword_extract_component(text))
            severities.append("Unknown")

    standardized = pd.DataFrame({
        "note_id": [f"{source_filename}-{i+1:04d}" for i in range(len(working))],
        "product_model": (
            working[meta_cols["product_model"]].fillna("Unknown").astype(str).values
            if meta_cols["product_model"] else "Unknown"
        ),
        "date": (
            pd.to_datetime(working[meta_cols["date"]], errors="coerce")
            .fillna(pd.Timestamp(datetime.today().date())).values
            if meta_cols["date"] else pd.Timestamp(datetime.today().date())
        ),
        "symptom_text": working["_clean_review"].values,
        "fix_text": "",
        "component": components,
        "severity": severities,
    })

    # Carry through rating/location/customer if present, as extra
    # reference columns. The existing pipeline ignores columns it
    # doesn't know about, so this can't break anything downstream.
    for standard_name in ("rating", "location", "customer"):
        if meta_cols.get(standard_name):
            standardized[standard_name] = working[meta_cols[standard_name]].values

    detection_info = {
        "input_type": "raw_reviews",
        "review_column": review_col,
        "metadata_columns_found": {k: v for k, v in meta_cols.items() if v},
        "rows_in": rows_in,
        "rows_after_cleaning": len(working),
        "rows_dropped": rows_dropped,
        "ai_extraction_used": ai_used,
    }
    return standardized, detection_info