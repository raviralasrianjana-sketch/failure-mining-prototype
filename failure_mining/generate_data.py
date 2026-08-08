"""
generate_data.py
-----------------
Creates a SYNTHETIC dataset of field return / service notes.

Why synthetic? Because in a real company this data would be sensitive
(customer info, product defects). For a hackathon demo, we fake it so the
whole pipeline can be shown end-to-end without needing real company data.

What this script does:
1. Defines a handful of "failure themes" (no power, overheating, etc.)
2. For each theme, writes several template sentences (so notes look human,
   not robotic/repeated).
3. Randomly mixes in some PII (fake names, phone numbers, emails) into a
   portion of the notes -- on purpose! This lets us prove our sanitizer
   (in pipeline.py) actually removes it.
4. Spreads the notes across 3 product models and 12 months, and deliberately
   makes "overheating" complaints trend upward over time for one model --
   so the trend chart in the dashboard has something interesting to show.

Output: data/service_notes.csv
"""

import csv
import random
from datetime import date, timedelta

random.seed(42)  # fixed seed -> same fake data every time you run this

# ---------------------------------------------------------------------------
# 1. Define failure themes and example phrasings for each.
#    In real data, notes would be free text typed by a technician.
#    We simulate that variety with multiple template sentences per theme.
# ---------------------------------------------------------------------------
THEMES = {
    "No Power": {
        "symptoms": [
            "Unit does not power on at all.",
            "Device completely dead, no response to power button.",
            "No power, LED does not light up when plugged in.",
            "Customer reports unit won't turn on even after charging overnight.",
            "Dead on arrival, no boot, no lights.",
        ],
        "fixes": [
            "Replaced power supply board.",
            "Replaced battery, unit powers on normally now.",
            "Found loose power connector, reseated and resolved.",
            "Replaced fuse on main board.",
        ],
    },
    "Overheating": {
        "symptoms": [
            "Device gets very hot to touch during normal use.",
            "Unit overheats and shuts down after 20 minutes of operation.",
            "Customer reports burning smell and excessive heat near vents.",
            "Thermal shutdown triggered repeatedly under light load.",
            "Fan not spinning, unit overheating quickly.",
        ],
        "fixes": [
            "Cleaned dust from heat sink and fan assembly.",
            "Replaced thermal paste on main processor.",
            "Replaced faulty cooling fan.",
            "Replaced temperature sensor, false trips resolved.",
        ],
    },
    "Display Fault": {
        "symptoms": [
            "Screen shows vertical lines across display.",
            "Display is completely blank but unit powers on.",
            "Touchscreen not responding in bottom half of screen.",
            "Flickering screen, gets worse when device is warm.",
            "Cracked display panel, screen partially visible.",
        ],
        "fixes": [
            "Replaced display panel assembly.",
            "Reseated display ribbon cable.",
            "Replaced touch digitizer.",
            "Replaced display driver IC on board.",
        ],
    },
    "Battery Drain": {
        "symptoms": [
            "Battery drains fully within 2 hours of light use.",
            "Device loses charge rapidly even in standby mode.",
            "Battery percentage drops from 100 to 0 in minutes.",
            "Customer reports battery does not hold charge overnight.",
        ],
        "fixes": [
            "Replaced battery cell, capacity restored.",
            "Updated firmware to fix power management bug.",
            "Replaced charging circuit board.",
        ],
    },
    "Connectivity Issue": {
        "symptoms": [
            "Unit repeatedly disconnects from WiFi network.",
            "Bluetooth pairing fails every time.",
            "Customer reports intermittent loss of network signal.",
            "Device cannot detect any nearby WiFi networks.",
        ],
        "fixes": [
            "Replaced WiFi/Bluetooth module.",
            "Updated network driver firmware.",
            "Reseated antenna connector.",
        ],
    },
    "Physical Damage": {
        "symptoms": [
            "Housing cracked near hinge, likely from a drop.",
            "Outer casing has visible dents and scratches.",
            "Broken latch, cover does not close properly.",
            "Customer reports unit was dropped, casing damaged.",
        ],
        "fixes": [
            "Replaced outer housing.",
            "Replaced broken hinge assembly.",
            "Replaced latch mechanism.",
        ],
    },
    "Software Crash": {
        "symptoms": [
            "Device freezes randomly during normal operation.",
            "App crashes repeatedly on startup.",
            "Unit stuck in boot loop after last update.",
            "Customer reports random restarts several times a day.",
        ],
        "fixes": [
            "Reflashed firmware to latest stable version.",
            "Factory reset resolved the crashing issue.",
            "Rolled back faulty software update.",
        ],
    },
    "Unusual Noise": {
        "symptoms": [
            "Loud grinding noise coming from internal fan.",
            "Clicking sound heard during startup.",
            "Rattling noise when device is moved.",
            "High pitched whining noise during operation.",
        ],
        "fixes": [
            "Replaced noisy fan assembly.",
            "Tightened loose internal bracket causing rattle.",
            "Replaced coil on power board causing whine.",
        ],
    },
}

PRODUCT_MODELS = ["Model-A100", "Model-B200", "Model-C300"]

# Fake names/phone/email snippets used ONLY to simulate PII appearing in
# free-text notes, so we can demonstrate sanitization removing it.
FAKE_NAMES = ["Rahul Sharma", "Priya Nair", "Vikram Rao", "Ananya Iyer", "Karthik Reddy"]
FAKE_PHONES = ["+91-98765-43210", "9845012345", "+91 90000 11122"]
FAKE_EMAILS = ["rahul.sharma@example.com", "priya.n@mail.com", "vikram.rao99@example.in"]


def maybe_add_pii(text: str) -> str:
    """Randomly sprinkle in fake PII to ~15% of notes (simulating messy real data)."""
    if random.random() < 0.15:
        choice = random.choice(["name", "phone", "email"])
        if choice == "name":
            text = f"Customer {random.choice(FAKE_NAMES)} reported: " + text
        elif choice == "phone":
            text = text + f" Contact: {random.choice(FAKE_PHONES)}."
        else:
            text = text + f" Reach out at {random.choice(FAKE_EMAILS)} for follow-up."
    return text


def random_date_in_last_year(bias_recent_for_theme=None, theme=None):
    """
    Pick a random date in the last 365 days.
    If bias_recent_for_theme is set and matches the theme, skew towards
    more recent dates -- this simulates a *rising trend* (e.g. Overheating
    complaints increasing in recent months), which makes the dashboard's
    trend chart meaningful instead of flat/random.
    """
    today = date.today()
    days_back = random.randint(0, 365)
    if bias_recent_for_theme and theme == bias_recent_for_theme:
        # Skew towards smaller days_back (i.e. more recent) using sqrt trick
        days_back = int(days_back * random.random() ** 2)
    return today - timedelta(days=days_back)


def generate_serial(model: str, idx: int) -> str:
    prefix = model.split("-")[1][:3]  # e.g. "A10" from "Model-A100"
    return f"SN-{prefix}-{idx:05d}"


def generate_dataset(n_notes=450, out_path="data/service_notes.csv"):
    rows = []
    for i in range(1, n_notes + 1):
        model = random.choice(PRODUCT_MODELS)
        theme = random.choice(list(THEMES.keys()))

        # Deliberately trend: Overheating complaints on Model-B200 rise recently
        rising_theme = "Overheating" if model == "Model-B200" else None
        note_date = random_date_in_last_year(bias_recent_for_theme=rising_theme, theme=theme)

        symptom = random.choice(THEMES[theme]["symptoms"])
        fix = random.choice(THEMES[theme]["fixes"])
        symptom = maybe_add_pii(symptom)

        # Optional structured tag - only present ~50% of the time (real data is inconsistent!)
        tag = None
        if random.random() < 0.5:
            tag = theme.lower().replace(" ", "_")

        rows.append({
            "note_id": f"RET-{i:05d}",
            "product_model": model,
            "serial_number": generate_serial(model, i),
            "date": note_date.isoformat(),
            "symptom_text": symptom,
            "fix_text": fix,
            "tag": tag or "",
            # true_theme is kept ONLY for us to sanity-check clustering quality.
            # A real dataset would NOT have this column -- the whole point of
            # clustering is that we DON'T know the theme in advance!
            "_true_theme_for_validation_only": theme,
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} synthetic service notes -> {out_path}")


if __name__ == "__main__":
    generate_dataset()
