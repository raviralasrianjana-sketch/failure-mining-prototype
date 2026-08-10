# 🔧 Field Returns Failure Mode Mining — Hackathon MVP

Turns messy, unstructured service/repair notes into clustered failure
themes, trend charts, and an actionable insights report — no manual
tagging required. Now with accounts, profile history, and an
icon-based navigation hub.

This matches the brief: **Service Notes → Top Failure Modes.**

---

## 1. What's in this folder

```
failure_mining/
├── generate_data.py         # Creates synthetic (fake) service notes -> data/service_notes.csv
├── pipeline.py               # The data science: sanitize -> vectorize -> cluster -> label -> trends
├── auth.py                   # Accounts (email/password + Google), profiles, history (SQLite)
├── app.py                     # Streamlit app: login -> upload -> icon hub -> sections -> history
├── requirements.txt           # Python packages needed
├── .streamlit/
│   └── secrets.toml.example    # Template for Google Sign-In config (copy -> secrets.toml)
├── data/
│   ├── service_notes.csv       # Generated synthetic dataset (450 notes)
│   └── app.db                  # SQLite database: users + history (created automatically)
├── reports/                    # Saved snapshot .docx reports, one per analysis run
└── README.md                  # You are here
```

We deliberately split **logic** (`pipeline.py`), **accounts**
(`auth.py`), and **UI** (`app.py`). This is good practice: you can
test/debug the data science by just running `python pipeline.py` in a
terminal, without touching the dashboard or login system at all.

---

## 2. How to run it

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Already done, but you can regenerate anytime)
python generate_data.py

# 3. (Optional) set up Google Sign-In -- see section 4 below.
#    Skip this if you only want email/password login for now.

# 4. Launch the app
streamlit run app.py
```

It'll open in your browser automatically (usually `http://localhost:8501`).
First screen: **sign up or log in.**

---

## 3. Accounts, profiles & history (new)

- **Starting the app now always begins with sign-in.** You either:
  - Enter an email + password to **sign up** (creates a profile) or
    **log in** (if you already have one), or
  - Click **Continue with Google** (needs a one-time setup — see below).
- A **profile** (name, email, sign-in method, created date) is stored the
  first time you sign up or sign in with Google. It's saved locally in
  `data/app.db` (SQLite — a single file, no server needed).
- Every time you run an analysis, it's logged to **your profile's
  History** (email + password hashed with bcrypt; Google accounts use no
  password at all). Open History anytime from the sidebar (🕒 History
  button) to see past runs and re-download their saved reports.
- Passwords are never stored as plain text — they're hashed with
  **bcrypt** before touching the database.

### Setting up Google Sign-In (optional)

Google Sign-In uses Streamlit's built-in `st.login()` (OpenID Connect).
Email/password login works with **zero** setup; only do this if you want
the Google button to work too.

1. Go to [Google Cloud Console](https://console.cloud.google.com/) →
   **APIs & Services → Credentials** → **Create Credentials → OAuth
   client ID** → Application type: **Web application**.
2. Add an **Authorized redirect URI**:
   `http://localhost:8501/oauth2callback` (for local dev — use your
   deployed URL + `/oauth2callback` in production).
3. Copy the generated **Client ID** and **Client Secret**.
4. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
   and fill in your Client ID/Secret and a random `cookie_secret` string.
5. Restart the app. The Google button will now work. If it's not
   configured, clicking it shows a friendly message instead of crashing
   — email/password login is unaffected either way.

`.streamlit/secrets.toml` is already in `.gitignore` — never commit real
credentials to GitHub.

---

## 4. Navigating the app (new: icon hub)

After you log in and upload + analyze a file, you land on a **hub page**
with six sections, each shown as an animated icon + topic name:

| Icon | Section |
|---|---|
| 📊 | Failure Mode Analysis Results (KPI overview) |
| 🧩 | Failure Themes (Overall) |
| 📈 | Trend Over Time by Theme |
| 🏭 | Failure Counts by Product Model |
| 💡 | Actionable Insights Summary |
| 📄 | Underlying Data & Report (download) |

Click any icon/name to open that section full-page; use **⬅️ Back to
Hub** to return. Sidebar filters (product model, date range) apply
across every section consistently.

---

## 5. Raw review data support (new)

You don't have to hand-format your input anymore. The uploader also
accepts **raw/unstructured review data** -- e.g. a CSV or Excel export
of Google Reviews, with arbitrary column names and no failure-specific
structure at all.

**How it decides what kind of file you gave it:** when a CSV/Excel/TSV/
JSON upload doesn't already have the `product_model`/`date`/
`symptom_text` columns the pipeline needs, it's automatically routed
through `review_preprocessor.py` instead of being rejected. That module:
- Finds the review/comment text column by name (`review`, `comment`,
  `feedback`, `description`, etc.) or, failing that, picks whichever
  text column reads like real sentences rather than short labels.
- Keeps any useful extra columns it finds (`model`, `date`, `rating`,
  `location`, `reviewer`) instead of throwing them away.
- Cleans up empty rows and exact duplicate reviews.
- Tags each review with a `component` (e.g. "Engine", "Brakes",
  "Battery") and `severity`, using a free built-in keyword dictionary --
  or, if you've configured an AI API key, a small LLM call for a nicer
  extraction (see below). Either way, it never invents a component that
  isn't actually mentioned in the review.

Once that's done, the reviews look exactly like any other service note
to the rest of the app -- clustering, trends, insights, and the Word
report all work identically, regardless of where the data came from.

**Optional: AI-enhanced extraction.** By default, no AI API key is
needed -- the keyword-based fallback just works. If you want richer
component/severity extraction, set an environment variable:
```
OPENAI_API_KEY=your-key-here
```
or add it to `.streamlit/secrets.toml`:
```toml
[openai]
api_key = "your-key-here"
```
Never commit real keys -- `secrets.toml` is already gitignored.

Two sample review files are included at `data/google_reviews_test.csv`
and `.xlsx` if you want to try this out immediately.

## 6. How the pipeline works (explain this in your demo!)

Think of it as an assembly line with stations:

### Station 1 — Sanitize (remove private info)
Real service notes often contain a customer's name, phone number, or
email typed in by a technician. Before anything else, regex patterns
find and redact these (`[REDACTED_EMAIL]`, etc.). Satisfies the brief's
"No PII; sanitize text fields" guardrail.

### Station 2 — Vectorize (turn text into numbers)
We use **TF-IDF** (Term Frequency–Inverse Document Frequency): each note
becomes a list of numbers where distinctive words (like *"overheating"*)
score high, and common words (like *"unit"*) score low.

> This is a lightweight stand-in for "embeddings" from the original
> brief. A production version could swap in neural sentence embeddings
> or Azure OpenAI embeddings — that's a one-function change
> (`vectorize_text()` in `pipeline.py`), everything downstream stays
> the same.

### Station 3 — Cluster (group similar notes)
**KMeans** groups similar notes automatically. We test a range of
cluster counts (`k = 4..8`) and pick the best **silhouette score**.

### Station 4 — Label (name each cluster)
Each cluster starts with a keyword-based label (top TF-IDF terms). In a
production system, an LLM (Azure OpenAI, as suggested in the brief)
could turn the keywords into a polished label — kept keyword-based here
so the demo needs zero API keys.

### Station 4b — Use optional structured tags (new)
If some notes carry a structured tag (e.g. `overheating`,
`no_power` — the brief's "optional structured tags" field), we check
whether tagged notes inside a cluster mostly agree on one tag. If they
do, we swap the raw keyword label for a clean name built from that tag
(e.g. `"Overheating"` instead of `"Overheats / Fan / Hot / Shuts"`), and
surface a **tag agreement %** in the Insights view as a QA signal. If
tags are sparse or disagree, we keep the keyword-based label — tags are
optional evidence, not a requirement.

### Station 5 — Trend & Insights
Once every note has a `theme`, we:
- Count notes per theme per month per product model → trend chart
- Sort themes by frequency → "top failure drivers"
- Pull 2–3 real (sanitized) example notes per theme, plus the most
  common **serial range** per theme (from the brief's `serial_range`
  metadata field) → supporting evidence

---

## 7. About the synthetic data

`generate_data.py` creates 450 fake service notes across 3 product
models and 8 realistic failure themes. It also:
- Randomly injects fake PII into ~15% of notes, so you can *prove* the
  sanitizer works in your demo (show a before/after).
- Randomly attaches a `tag` to ~50% of notes and a `serial_number` to
  every note, matching the brief's optional metadata fields.
- Deliberately makes **Overheating complaints rise recently on
  Model-B200**, so the trend chart has a real story to tell.

To regenerate with different randomness, edit `random.seed(42)` at the
top of `generate_data.py`, or increase `n_notes` for a bigger dataset.

---

## 8. Honest limitations (good to mention — judges like self-awareness!)

- TF-IDF clustering picks up on *word overlap*, not deep meaning — neural
  embeddings would help here.
- The PII sanitizer uses simple regex, tuned to this demo's fake data
  patterns; production would need a proper NER model for robustness on
  real names/addresses.
- Accounts/history use a local SQLite file and Streamlit session state —
  fine for a single-machine demo, but not multi-device or highly
  concurrent; a production version would use a real auth provider and
  a hosted database.
- Cluster labels are keyword lists (or a matched tag) — an LLM call
  would make them read even better.
- This is explicitly an **MVP** — matches the brief's note: "MVP scope
  only; production hardening not required."

---

## 9. Natural next steps (if you have hackathon time left)

1. Swap `vectorize_text()` for real embeddings (sentence-transformers or
   Azure OpenAI) — biggest quality upgrade for least effort.
2. Send each cluster's top keywords to an LLM to generate a clean label
   and a 1-sentence root-cause hypothesis.
3. Add a feedback loop: let a technician mark "wrong cluster" on a note,
   and use that to fine-tune / re-cluster.
4. Add anomaly detection to flag sudden spikes in a theme automatically.
5. Move accounts/history from SQLite to a hosted database and add real
   session cookies so login persists across browser restarts.
