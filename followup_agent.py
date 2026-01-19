# app.py — NephroBridge (Product-grade, Hackathon-ready)
# ------------------------------------------------------------
# A calm, patient-first kidney clarity tool powered by Google HAI-DEF models.
# - Not a diagnostic system
# - Not a medical decision-maker
# - A medical-aware explanation + sense-making layer
#
# This app orchestrates 3 narrow agents:
# 1) Lab Interpretation Agent
# 2) Medication Change Explainer
# 3) Follow-up & Open-Loops Tracker
#
# It supports two model backends:
# - MedGemma (HAI-DEF) for hackathon compliance (recommended)
# - Gemma-2B for lightweight local/dev fallback
# ------------------------------------------------------------

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st

from lab_agent_v1 import run_lab_agent_from_timeline
from medication_agent import run_medication_agent
from followup_agent import run_followup_agent


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "DATA"
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = DATA_DIR / "patient_timeline_user.json"


# Friendly UI mapping -> internal stage keys
STAGES = {
    "I recently had a kidney transplant": "post_transplant",
    "I have advanced kidney disease (not on dialysis)": "advanced_ckd",
    "I’m currently on dialysis": "dialysis",
}

MODEL_CHOICES = {
    "MedGemma (HAI-DEF, healthcare-specialized)": True,
    "Gemma-2B (lightweight fallback for local/dev)": False,
}


# ------------------------------------------------------------
# PAGE THEME (calm, supportive product framing)
# ------------------------------------------------------------
st.set_page_config(
    page_title="NephroBridge",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
/* Make headings feel less “developer demo” and more “product” */
h1 { margin-bottom: 0.15rem; }
.small-note { color: rgba(49,51,63,0.65); font-size: 0.92rem; line-height: 1.35; }
.soft-card {
  background: rgba(250,250,255,0.85);
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 16px;
  padding: 14px 14px;
}
.pill {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 999px;
  border: 1px solid rgba(0,0,0,0.10);
  background: rgba(255,255,255,0.65);
  font-size: 0.85rem;
  margin-right: 6px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# HEADER (Human-first, non-scary framing)
# ------------------------------------------------------------
st.title("NephroBridge")
st.write("A calm explanation tool for kidney labs, medication changes, and what comes next.")

st.markdown(
    """
<div class="soft-card">
  <span class="pill">Not a diagnosis tool</span>
  <span class="pill">No medical advice</span>
  <span class="pill">Patient clarity</span>
  <div class="small-note" style="margin-top:10px;">
    NephroBridge helps reduce confusion and anxiety when lab results change,
    medications get adjusted, or follow-up plans feel unclear.
    It uses <b>Google’s open HAI-DEF models (MedGemma)</b> to generate
    <b>patient-friendly explanations</b> — without making clinical decisions.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.divider()


# ------------------------------------------------------------
# HELPERS — timeline creation & normalization
# ------------------------------------------------------------
def normalize_labs_table(df: pd.DataFrame) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """Make CSV flexible: accept several column name variants."""
    col_map = {c.lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in col_map:
                return col_map[n]
        return None

    date_col = pick("date", "lab_date", "timestamp", "collected_date")
    name_col = pick("lab", "lab_name", "test", "test_name")
    value_col = pick("value", "result", "lab_value")
    unit_col = pick("unit", "units")
    ref_col = pick("reference_range", "ref_range", "range", "ref")

    required = [date_col, name_col, value_col]
    if any(x is None for x in required):
        return None, "CSV must contain at least columns for: date, lab_name, value (names can vary like lab/test and result/value)."

    labs: List[Dict] = []
    for _, row in df.iterrows():
        labs.append(
            {
                "date": str(row[date_col]).strip(),
                "lab_name": str(row[name_col]).strip(),
                "value": str(row[value_col]).strip(),
                "unit": "" if unit_col is None else str(row[unit_col]).strip(),
                "reference_range": "" if ref_col is None else str(row[ref_col]).strip(),
            }
        )
    return labs, None


def lines_to_list(text: str) -> List[str]:
    return [x.strip() for x in (text or "").splitlines() if x.strip()]


def build_timeline_json(
    stage_key: str,
    reason: str,
    labs: List[Dict],
    meds_before: List[str],
    meds_after: List[str],
    followups: List[str],
    pending_labs: List[str],
) -> Dict:
    return {
        "kidney_journey_stage": stage_key,
        "recent_hospital_stay": {"reason_for_admission": reason.strip()} if reason.strip() else {},
        "labs_over_time": labs,
        "medications_before": meds_before,
        "medications_after": meds_after,
        "followup_appointments": followups,
        "pending_labs": pending_labs,
    }


def validate_minimum(timeline: Dict) -> Optional[str]:
    labs = timeline.get("labs_over_time", [])
    if not isinstance(labs, list) or len(labs) == 0:
        return "Please add at least one lab result (date, lab name, value)."
    # minimal required keys
    for i, r in enumerate(labs, start=1):
        if not str(r.get("date", "")).strip() or not str(r.get("lab_name", "")).strip() or not str(r.get("value", "")).strip():
            return f"Lab row {i} is missing date, lab name, or value."
    return None


# ------------------------------------------------------------
# STEP 1 — Journey context + model choice (kept gentle)
# ------------------------------------------------------------
st.markdown("## Step 1 — Where are you in your kidney journey?")
stage_label = st.radio(
    "Choose what fits best today:",
    list(STAGES.keys()),
    index=0,
)
stage_key = STAGES[stage_label]

st.markdown("## Step 1a — Choose the AI engine")
model_label = st.radio(
    "For the hackathon demo, MedGemma is recommended.",
    list(MODEL_CHOICES.keys()),
    index=0,
)
use_medgemma = MODEL_CHOICES[model_label]
os.environ["USE_MEDGEMMA"] = "true" if use_medgemma else "false"

st.caption(
    "MedGemma is part of Google’s Health AI Developer Foundations (HAI-DEF). "
    "NephroBridge uses it to produce cautious, structured explanations — not decisions."
)

st.divider()


# ------------------------------------------------------------
# STEP 2 — Patient-friendly input (form OR upload)
# ------------------------------------------------------------
st.markdown("## Step 2 — Add your recent information")
st.write("You can type what you see on your lab report, or upload a file. Keep it simple — NephroBridge will structure it for the agents.")

mode = st.radio(
    "How would you like to add your info?",
    ["Fill a simple form", "Upload a file (CSV or JSON)"],
    horizontal=True,
)

timeline: Optional[Dict] = None
error: Optional[str] = None


if mode == "Fill a simple form":
    with st.form("patient_form", clear_on_submit=False):
        st.markdown("### Why were you seen recently? (optional)")
        reason = st.text_input(
            "Example: routine follow-up, dehydration, swelling, lab review",
            placeholder="A short phrase is enough",
        )

        st.markdown("### Your lab results")
        st.caption("Add 1–8 results. Example: Creatinine 1.6 mg/dL on 2026-01-08.")
        lab_rows = st.data_editor(
            [{"date": "", "lab_name": "", "value": "", "unit": "", "reference_range": ""}],
            num_rows="dynamic",
            use_container_width=True,
        )

        st.markdown("### Medication changes (optional)")
        st.caption("If something changed, list before vs after. One item per line.")
        c1, c2 = st.columns(2)
        with c1:
            meds_before_text = st.text_area("Before", height=120, placeholder="Tacrolimus 1mg\nPrednisone 5mg")
        with c2:
            meds_after_text = st.text_area("After", height=120, placeholder="Tacrolimus 2mg\nPrednisone 5mg")

        st.markdown("### Follow-ups & open loops (optional)")
        followups_text = st.text_area(
            "Appointments / instructions (one per line)",
            height=100,
            placeholder="Nephrology follow-up in 1 week\nRepeat labs on Monday",
        )
        pending_labs_text = st.text_area(
            "Pending labs / results (one per line)",
            height=80,
            placeholder="Tacrolimus level\nUrine culture",
        )

        submitted = st.form_submit_button("Continue →")

    if submitted:
        labs: List[Dict] = []
        for r in lab_rows:
            if str(r.get("date", "")).strip() and str(r.get("lab_name", "")).strip() and str(r.get("value", "")).strip():
                labs.append(
                    {
                        "date": str(r.get("date", "")).strip(),
                        "lab_name": str(r.get("lab_name", "")).strip(),
                        "value": str(r.get("value", "")).strip(),
                        "unit": str(r.get("unit", "")).strip(),
                        "reference_range": str(r.get("reference_range", "")).strip(),
                    }
                )

        meds_before = lines_to_list(meds_before_text)
        meds_after = lines_to_list(meds_after_text)
        followups = lines_to_list(followups_text)
        pending_labs = lines_to_list(pending_labs_text)

        timeline = build_timeline_json(stage_key, reason, labs, meds_before, meds_after, followups, pending_labs)
        error = validate_minimum(timeline)

else:
    st.markdown("### Upload a file")
    st.caption("Accepted: CSV of labs OR a NephroBridge JSON timeline. We'll convert and validate it.")
    uploaded = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])

    st.markdown("### (Optional) Why were you seen recently?")
    reason = st.text_input("Example: dehydration, routine follow-up, lab review")

    if uploaded is not None:
        try:
            suffix = uploaded.name.lower().split(".")[-1]
            if suffix == "json":
                raw = json.load(uploaded)
                if isinstance(raw, dict) and "labs_over_time" in raw:
                    # coerce stage to match UI selection
                    raw["kidney_journey_stage"] = stage_key
                    if reason.strip():
                        raw["recent_hospital_stay"] = {"reason_for_admission": reason.strip()}
                    timeline = raw
                else:
                    error = "JSON uploaded, but it doesn't look like a NephroBridge timeline. Use the template shown below."
            else:
                df = pd.read_csv(uploaded)
                labs, err = normalize_labs_table(df)
                if err:
                    error = err
                else:
                    timeline = build_timeline_json(stage_key, reason, labs, [], [], [], [])
            if timeline:
                error = validate_minimum(timeline)
        except Exception as e:
            error = f"Could not read file: {e}"


if error:
    st.error(error)


# ------------------------------------------------------------
# STEP 2b — Helpful template download (for hackathon realism)
# ------------------------------------------------------------
with st.expander("Need a template? (CSV + JSON examples)"):
    st.markdown("### CSV template (labs)")
    csv_template = pd.DataFrame(
        [
            {
                "date": "2026-01-08",
                "lab_name": "Creatinine",
                "value": "1.6",
                "unit": "mg/dL",
                "reference_range": "0.6–1.3",
            }
        ]
    )
    st.dataframe(csv_template, use_container_width=True)

    st.markdown("### JSON template (full timeline)")
    template = {
        "kidney_journey_stage": "post_transplant | advanced_ckd | dialysis",
        "recent_hospital_stay": {"reason_for_admission": "optional"},
        "labs_over_time": [
            {"date": "YYYY-MM-DD", "lab_name": "Creatinine", "value": "1.6", "unit": "mg/dL", "reference_range": "0.6–1.3"}
        ],
        "medications_before": ["Tacrolimus 1mg (example)"],
        "medications_after": ["Tacrolimus 2mg (example)"],
        "followup_appointments": ["Nephrology follow-up in 1 week"],
        "pending_labs": ["Tacrolimus level"],
    }
    st.code(json.dumps(template, indent=2), language="json")


# ------------------------------------------------------------
# STEP 3 — One calm action + results
# ------------------------------------------------------------
if timeline:
    st.divider()

    st.markdown("## Step 3 — Get a calm explanation")
    st.write("When you’re ready, NephroBridge will explain what changed and what to expect next — without giving medical advice.")

    # Save timeline optionally (useful for reproducibility)
    cA, cB = st.columns([1, 1])
    with cA:
        if st.button("Save this timeline", use_container_width=True):
            OUTPUT_PATH.write_text(json.dumps(timeline, indent=2))
            st.success(f"Saved to {OUTPUT_PATH}")
    with cB:
        st.download_button(
            "Download timeline JSON",
            data=json.dumps(timeline, indent=2).encode("utf-8"),
            file_name="patient_timeline_user.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()

    # PRIMARY CTA
    if st.button("Help me understand what’s going on", type="primary", use_container_width=True):
        # Make outputs feel “guided”
        st.markdown(
            """
<div class="soft-card">
  <div class="small-note">
    <b>Safety note:</b> These explanations are for clarity and preparation — not diagnosis.
    If you feel unwell or your symptoms are worsening, contact your care team.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.write("")

        with st.spinner("Reading your information and generating explanations…"):
            t0 = time.time()
            lab_text = run_lab_agent_from_timeline(timeline)
            med_text = run_medication_agent(timeline)
            follow_text = run_followup_agent(timeline)
            elapsed = time.time() - t0

        st.caption(f"Generated in {elapsed:.1f}s • Model: {'MedGemma' if use_medgemma else 'Gemma-2B'}")

        st.divider()

        tab1, tab2, tab3 = st.tabs(
            ["🧪 What changed in my labs", "💊 Why my medications changed", "🧾 What I should expect next"]
        )

        with tab1:
            st.markdown(lab_text)

        with tab2:
            st.markdown(med_text)

        with tab3:
            st.markdown(follow_text)

        with st.expander("See the structured timeline NephroBridge used (for transparency)"):
            st.json(timeline)

else:
    st.info("Add your information in Step 2 to continue.")
