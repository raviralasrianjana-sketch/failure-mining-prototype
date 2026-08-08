"""
pipeline.py
-----------
The "brain" of the project. This file has NO dashboard code in it --
it's pure data processing so it's easy to test and reason about on its own.

Stages (matches the flow in the problem statement):
  1. load_data      -> read the CSV
  2. sanitize_text   -> remove PII (no names/phone/email should reach the model)
  3. vectorize_text  -> turn each note into numbers (TF-IDF "embedding")
  4. cluster_notes    -> group similar notes together (KMeans)
  5. label_clusters   -> give each cluster a human-readable name (top keywords)
  6. build_trend_table -> count failures per model per month
  7. build_insights    -> plain-English summary of the top issues

WHY TF-IDF INSTEAD OF A NEURAL EMBEDDING MODEL?
Neural sentence embeddings (e.g. sentence-transformers, OpenAI embeddings)
would work too and are mentioned in the original brief. But they need a
model download / API key and a GPU helps. For a hackathon MVP, TF-IDF:
  - runs instantly, offline, no API key
  - is easy to explain in a demo ("it counts important words in each note
    and compares notes based on which words they share")
  - is a legitimate, commonly used baseline for text clustering

If you have time near the end of the hackathon, swapping in
sentence-transformers or Azure OpenAI embeddings is a small, isolated change
-- only vectorize_text() needs to change. Everything downstream (clustering,
labeling, trends) stays the same. That's a good "future work" slide!
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# 2. SANITIZE (remove PII) -- a Constraint & Guardrail from the brief
# ---------------------------------------------------------------------------
# Simple regex patterns. Not perfect (real PII scrubbing is a deep topic),
# but demonstrates the *principle*: no email/phone/name should reach the
# model or the dashboard.
EMAIL_RE = re.compile(r"[\w\.\-]+@[\w\.\-]+\.\w+")
PHONE_RE = re.compile(r"(\+?\d[\d\-\s]{8,}\d)")
# Very simple "Customer <First Last>" name pattern used by our fake data.
NAME_RE = re.compile(r"Customer [A-Z][a-z]+ [A-Z][a-z]+")


def sanitize_text(text: str) -> str:
    """Redact emails, phone numbers, and simple name patterns from a note."""
    if not isinstance(text, str):
        return ""
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = NAME_RE.sub("Customer [REDACTED_NAME]", text)
    return text


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["symptom_text_clean"] = df["symptom_text"].apply(sanitize_text)
    df["fix_text_clean"] = df["fix_text"].apply(sanitize_text)
    return df


# ---------------------------------------------------------------------------
# 3. VECTORIZE (turn text into numbers so a computer can compare notes)
# ---------------------------------------------------------------------------
def vectorize_text(texts):
    """
    TF-IDF = Term Frequency - Inverse Document Frequency.
    In plain English: it scores each word in a note by how important that
    word seems to be -- common words like "the"/"unit" get a low score,
    while distinctive words like "overheating" or "cracked" get a high
    score. Each note becomes a vector (a long list of numbers), and notes
    with similar important words end up as similar vectors.
    """
    # Custom stopwords: generic service-note boilerplate that shows up in
    # almost every note regardless of failure type ("customer", "reports",
    # "device", "unit"...). Left in, these words would dominate every
    # cluster centroid and drown out the words that actually distinguish
    # one failure type from another.
    custom_stopwords = [
        "customer", "reports", "report", "reported", "device", "unit",
        "units", "issue", "problem", "time", "normal",
    ]
    stopwords = list(TfidfVectorizer(stop_words="english").get_stop_words()) + custom_stopwords

    vectorizer = TfidfVectorizer(
        stop_words=stopwords,
        ngram_range=(1, 2),  # capture two-word phrases too, e.g. "no power"
        min_df=2,            # ignore words that appear in only 1 note (too rare)
        max_df=0.4,          # ignore words that appear in >40% of notes (too common)
        max_features=500,
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer


# ---------------------------------------------------------------------------
# 4. CLUSTER (group similar notes together)
# ---------------------------------------------------------------------------
def choose_best_k(matrix, k_range=range(4, 9)):
    """
    KMeans needs to be told how many clusters (k) to find. We don't know
    the "true" number of failure themes in real data, so we try a range of
    k values and pick the one with the best silhouette score (a measure of
    how well-separated the clusters are -- higher is better, max is 1.0).
    """
    best_k, best_score = k_range[0], -1
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(matrix)
        score = silhouette_score(matrix, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def cluster_notes(matrix, n_clusters=None):
    if n_clusters is None:
        n_clusters = choose_best_k(matrix)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(matrix)
    return labels, km, n_clusters


# ---------------------------------------------------------------------------
# 5. LABEL CLUSTERS (turn "Cluster 3" into "Overheating / Thermal Shutdown")
# ---------------------------------------------------------------------------
def label_clusters(km, vectorizer, top_n=4):
    """
    For each cluster, look at its centroid (the "average" note in that
    cluster) and pull out the words/phrases with the highest TF-IDF weight.
    Those words become the cluster's auto-generated label.

    NOTE: In the original brief, this step was suggested to use Azure
    OpenAI to write a nicer label. That's a great upgrade -- just take
    these top keywords and ask an LLM: "these words describe a product
    failure theme, give it a short name." We keep it keyword-based here
    so the MVP needs no API key and is instant.
    """
    terms = vectorizer.get_feature_names_out()
    labels = {}
    for cluster_id, centroid in enumerate(km.cluster_centers_):
        top_indices = centroid.argsort()[::-1][:top_n]
        top_terms = [terms[i] for i in top_indices]
        labels[cluster_id] = " / ".join(top_terms).title()
    return labels


# ---------------------------------------------------------------------------
# FULL PIPELINE (glues steps 1-5 together)
# ---------------------------------------------------------------------------
def run_pipeline(csv_path: str, n_clusters=None):
    df = load_data(csv_path)
    df = sanitize_dataframe(df)

    matrix, vectorizer = vectorize_text(df["symptom_text_clean"])
    cluster_ids, km, k_used = cluster_notes(matrix, n_clusters=n_clusters)
    cluster_labels = label_clusters(km, vectorizer)

    df["cluster_id"] = cluster_ids
    df["theme"] = df["cluster_id"].map(cluster_labels)

    return df, cluster_labels, k_used


# ---------------------------------------------------------------------------
# 6. TREND TABLE (counts per model per month per theme)
# ---------------------------------------------------------------------------
def build_trend_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M").astype(str)
    trend = (
        df.groupby(["month", "product_model", "theme"])
        .size()
        .reset_index(name="count")
    )
    return trend


# ---------------------------------------------------------------------------
# 7. INSIGHTS SUMMARY (plain-English top drivers + examples)
# ---------------------------------------------------------------------------
def build_insights(df: pd.DataFrame, top_n=5, examples_per_theme=3):
    theme_counts = df["theme"].value_counts().head(top_n)

    insights = []
    for theme, count in theme_counts.items():
        pct = round(100 * count / len(df), 1)
        examples = (
            df[df["theme"] == theme]["symptom_text_clean"]
            .drop_duplicates()
            .head(examples_per_theme)
            .tolist()
        )
        top_model = df[df["theme"] == theme]["product_model"].value_counts().idxmax()
        insights.append({
            "theme": theme,
            "count": int(count),
            "pct_of_total": pct,
            "most_affected_model": top_model,
            "examples": examples,
        })
    return insights


if __name__ == "__main__":
    # Quick manual test: run the whole pipeline and print a summary.
    df, labels, k = run_pipeline("data/service_notes.csv")
    print(f"Chosen number of clusters (k): {k}")
    print("\nDiscovered themes:")
    for cid, label in labels.items():
        count = (df["cluster_id"] == cid).sum()
        print(f"  Cluster {cid}: {label}  ({count} notes)")

    print("\nSample sanitized notes (check PII removed):")
    pii_mask = df["symptom_text"].str.contains(r"@|\+91|Customer [A-Z]", regex=True, na=False)
    for _, sample in df[pii_mask].head(3).iterrows():
        print("  BEFORE:", sample["symptom_text"])
        print("  AFTER :", sample["symptom_text_clean"])
        print()
