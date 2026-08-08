# 🔧 Field Returns Failure Mode Mining — Hackathon MVP

Turns messy, unstructured service/repair notes into clustered failure
themes, trend charts, and an actionable insights report — no manual
tagging required.

This matches the brief: **Service Notes → Top Failure Modes.**

---

## 1. What's in this folder

```
failure_mining/
├── generate_data.py   # Creates synthetic (fake) service notes -> data/service_notes.csv
├── pipeline.py         # The data science: sanitize -> vectorize -> cluster -> label -> trends
├── app.py               # Streamlit dashboard (the UI)
├── requirements.txt     # Python packages needed
├── data/
│   └── service_notes.csv   # Generated synthetic dataset (450 notes)
└── README.md             # You are here
```

We deliberately split **logic** (`pipeline.py`) from **UI** (`app.py`).
This is good practice: you can test/debug the data science by just running
`python pipeline.py` in a terminal, without touching the dashboard at all.

---

## 2. How to run it

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Already done, but you can regenerate anytime)
python generate_data.py

# 3. Launch the dashboard
streamlit run app.py
```

It'll open in your browser automatically (usually `http://localhost:8501`).

---

## 3. How the pipeline works (explain this in your demo!)

Think of it as an assembly line with 5 stations:

### Station 1 — Sanitize (remove private info)
Real service notes often contain a customer's name, phone number, or
email typed in by a technician. Before we do anything else, regex
patterns find and redact these (`[REDACTED_EMAIL]`, etc.). This satisfies
the brief's "No PII; sanitize text fields" guardrail.

### Station 2 — Vectorize (turn text into numbers)
Computers can't "read" — they need numbers. We use **TF-IDF**
(Term Frequency–Inverse Document Frequency): each note becomes a long
list of numbers, where distinctive words (like *"overheating"* or
*"cracked"*) get a high score, and common/boring words (like *"unit"* or
*"customer"*) get a low score. Notes that share important words end up
with similar number-lists (vectors).

> This is a lightweight stand-in for "embeddings" from the original
> brief. A production version could swap in neural sentence embeddings
> (e.g. `sentence-transformers`) or Azure OpenAI embeddings here — that's
> a one-function change (`vectorize_text()` in `pipeline.py`), everything
> downstream stays the same.

### Station 3 — Cluster (group similar notes)
We use **KMeans** clustering on those number-vectors to automatically
group similar notes together — notes about overheating end up in one
group, notes about cracked screens in another, etc. We don't tell it in
advance how many groups exist; we test a range (`k = 4..8`) and pick
the one with the best **silhouette score** (a standard measure of how
well-separated clusters are).

### Station 4 — Label (name each cluster)
Each cluster gets an automatic, human-readable label built from its
most distinctive keywords (e.g. *"Screen / Display / Cracked"*). In a
production system this is exactly where you'd plug in an LLM (Azure
OpenAI, as suggested in the brief) to turn the keyword list into a
polished one-line theme name — we kept it keyword-based here so the demo
needs **zero API keys** and runs instantly offline.

### Station 5 — Trend & Insights
Once every note has a `theme`, we can:
- Count notes per theme per month per product model → trend chart
- Sort themes by frequency → "top failure drivers"
- Pull 2–3 real (sanitized) example notes per theme → supporting evidence

---

## 4. About the synthetic data

`generate_data.py` creates 450 fake service notes across 3 product
models and 8 realistic failure themes (No Power, Overheating, Display
Fault, Battery Drain, Connectivity, Physical Damage, Software Crash,
Unusual Noise). It also:
- Randomly injects fake PII into ~15% of notes, so you can *prove* the
  sanitizer works in your demo (show a before/after).
- Deliberately makes **Overheating complaints rise recently on
  Model-B200**, so the trend chart has a real story to tell — e.g.
  "Overheating on Model-B200 is trending up, recommend checking thermal
  design on the next revision."

To regenerate with different randomness, edit `random.seed(42)` at the
top of `generate_data.py`, or increase `n_notes` for a bigger dataset.

---

## 5. Dashboard walkthrough (for your demo script)

1. **KPI row** — total notes, themes discovered, top failure mode, models covered.
2. **Failure Themes chart** — which failure modes are most common overall.
3. **Trend Over Time** — is a theme getting worse or better, month by month.
4. **Failure Counts by Product Model** — which model has which problems.
5. **Actionable Insights** — expandable cards, one per theme, with real
   example notes as evidence (so engineers can trust the auto-labeling).
6. **Data table + CSV export** — for anyone who wants to dig into the raw
   (sanitized) records.

---

## 6. Honest limitations (good to mention — judges like self-awareness!)

- TF-IDF clustering picks up on *word overlap*, not deep meaning — two
  notes describing the same failure in very different words might not
  cluster together. Neural embeddings would help here.
- The PII sanitizer here uses simple regex; production would need a
  proper NER (Named Entity Recognition) model for robustness.
- Cluster labels are keyword lists, not polished sentences — swapping in
  an LLM call would make them read better.
- This is explicitly an **MVP** — matches the brief's note: "MVP scope
  only; production hardening not required."

---

## 7. Natural next steps (if you have hackathon time left)

1. Swap `vectorize_text()` for real embeddings (sentence-transformers or
   Azure OpenAI) — biggest quality upgrade for least effort.
2. Send each cluster's top keywords to an LLM to generate a clean label
   and a 1-sentence root-cause hypothesis.
3. Add a feedback loop: let a technician mark "wrong cluster" on a note,
   and use that to fine-tune / re-cluster.
4. Add anomaly detection to flag sudden spikes in a theme automatically.
