# app.py
import json
from pathlib import Path
import streamlit as st
import pandas as pd

DATA_DIR = Path("DATA")
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = DATA_DIR / "patient_timeline_user.json"

STAGES = {
    "Post-transplant": "post_transplant",
    "Advanced CKD (not on dialysis)": "advanced_ckd",
    "Dialysis": "dialysis",
}

st.set_page_config(page_title="NephroBridge", page_icon="🧠", layout="wide")

st.title("NephroBridge")
st.caption("A calm, patient-first explanation tool for kidney labs, meds, and follow-ups.")

st.markdown("### Step 1 — Tell us where you are in your kidney journey")
stage_label = st.radio(
    "Choose one:",
    list(STAGES.keys()),
    horizontal=True
)
stage = STAGES[stage_label]

st.divider()
st.markdown("### Step 2 — Add your information")
mode = st.radio(
    "How would you like to provide your info?",
    ["Fill a simple form", "Upload a file (CSV or JSON)"],
    horizontal=True
)

def normalize_labs_table(df: pd.DataFrame):
    """Make CSV flexible: accept several column name variants."""
    col_map = {c.lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in col_map:
                return col_map[n]
        return None

    date_col = pick("date", "lab_date", "timestamp")
    name_col = pick("lab", "lab_name", "test", "test_name")
    value_col = pick("value", "result", "lab_value")
    unit_col = pick("unit", "units")
    ref_col = pick("reference_range", "ref_range", "range")

    required = [date_col, name_col, value_col]
    if any(x is None for x in required):
        return None, "CSV must contain at least: date, lab_name, value (columns can be named date/lab/value too)."

    labs = []
    for _, row in df.iterrows():
        labs.append({
            "date": str(row[date_col]),
            "lab_name": str(row[name_col]),
            "value": str(row[value_col]),
            "unit": "" if unit_col is None else str(row[unit_col]),
            "reference_range": "" if ref_col is None else str(row[ref_col]),
        })
    return labs, None

def build_timeline_json(stage_key: str, reason: str, labs: list, meds_before: list, meds_after: list, followups: list, pending_labs: list):
    return {
        "kidney_journey_stage": stage_key,
        "recent_hospital_stay": {
            "reason_for_admission": reason.strip()
        } if reason.strip() else {},
        "labs_over_time": labs,
        "medications_before": meds_before,
        "medications_after": meds_after,
        "followup_appointments": followups,
        "pending_labs": pending_labs,
    }

timeline = None
error = None

if mode == "Fill a simple form":
    with st.form("patient_form"):
        st.markdown("#### Why were you recently seen in the hospital or clinic? (optional)")
        reason = st.text_input("Example: dehydration, high creatinine, swelling, routine transplant follow-up")

        st.markdown("#### Labs (add 1–6 results)")
        st.caption("Keep it simple. Example: Creatinine 1.6 mg/dL on 2026-01-08.")
        lab_rows = st.data_editor(
            [
                {"date": "", "lab_name": "", "value": "", "unit": "", "reference_range": ""},
            ],
            num_rows="dynamic",
            use_container_width=True
        )

        st.markdown("#### Medications (optional)")
        col1, col2 = st.columns(2)
        with col1:
            meds_before_text = st.text_area("Before (one per line)", height=120, placeholder="Tacrolimus 1mg\nPrednisone 5mg")
        with col2:
            meds_after_text = st.text_area("After (one per line)", height=120, placeholder="Tacrolimus 2mg\nPrednisone 5mg")

        st.markdown("#### Follow-ups & open loops (optional)")
        followups_text = st.text_area("Upcoming appointments / instructions (one per line)", height=100,
                                      placeholder="Nephrology follow-up in 1 week\nRepeat labs on Monday")
        pending_labs_text = st.text_area("Pending labs / results (one per line)", height=80,
                                         placeholder="Tacrolimus level\nUrine culture")

        submitted = st.form_submit_button("Create my timeline")

    if submitted:
        labs = []
        for r in lab_rows:
            if str(r.get("date", "")).strip() and str(r.get("lab_name", "")).strip() and str(r.get("value", "")).strip():
                labs.append({
                    "date": str(r.get("date", "")).strip(),
                    "lab_name": str(r.get("lab_name", "")).strip(),
                    "value": str(r.get("value", "")).strip(),
                    "unit": str(r.get("unit", "")).strip(),
                    "reference_range": str(r.get("reference_range", "")).strip(),
                })

        if len(labs) == 0:
            error = "Please add at least one lab result (date, lab name, value)."
        else:
            meds_before = [x.strip() for x in meds_before_text.splitlines() if x.strip()]
            meds_after = [x.strip() for x in meds_after_text.splitlines() if x.strip()]
            followups = [x.strip() for x in followups_text.splitlines() if x.strip()]
            pending_labs = [x.strip() for x in pending_labs_text.splitlines() if x.strip()]

            timeline = build_timeline_json(stage, reason, labs, meds_before, meds_after, followups, pending_labs)

else:
    st.markdown("#### Upload a file")
    st.caption("Accepted: CSV of labs OR JSON timeline. We'll convert it into the NephroBridge timeline format.")

    uploaded = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])

    st.markdown("#### (Optional) Why were you seen in hospital/clinic?")
    reason = st.text_input("Example: dehydration, high creatinine, swelling")

    if uploaded is not None:
        suffix = uploaded.name.lower().split(".")[-1]

        try:
            if suffix == "json":
                raw = json.load(uploaded)
                # If it already looks like our timeline, accept it.
                if isinstance(raw, dict) and "labs_over_time" in raw and "kidney_journey_stage" in raw:
                    timeline = raw
                    # override stage from UI if user selected one
                    timeline["kidney_journey_stage"] = stage
                    if reason.strip():
                        timeline["recent_hospital_stay"] = {"reason_for_admission": reason.strip()}
                else:
                    error = "JSON uploaded, but it doesn't look like a NephroBridge timeline. Use the template below."
            else:
                df = pd.read_csv(uploaded)
                labs, err = normalize_labs_table(df)
                if err:
                    error = err
                else:
                    timeline = build_timeline_json(stage, reason, labs, [], [], [], [])
        except Exception as e:
            error = f"Could not read file: {e}"

st.divider()

if error:
    st.error(error)

if timeline:
    st.success("Timeline created ✅")
    colA, colB = st.columns([2, 1])

    with colA:
        st.markdown("### Preview (this is what the agents will read)")
        st.json(timeline)

    with colB:
        st.markdown("### Save")
        if st.button("Save as DATA/patient_timeline_user.json"):
            OUTPUT_PATH.write_text(json.dumps(timeline, indent=2))
            st.success(f"Saved to {OUTPUT_PATH}")

    st.markdown("### Template you can copy (JSON)")
    template = {
        "kidney_journey_stage": "post_transplant | advanced_ckd | dialysis",
        "recent_hospital_stay": {"reason_for_admission": "optional"},
        "labs_over_time": [
            {"date": "YYYY-MM-DD", "lab_name": "Creatinine", "value": "1.6", "unit": "mg/dL", "reference_range": "0.6–1.3"}
        ],
        "medications_before": ["optional list"],
        "medications_after": ["optional list"],
        "followup_appointments": ["optional list"],
        "pending_labs": ["optional list"],
    }
    st.code(json.dumps(template, indent=2), language="json")
else:
    st.info("Fill the form or upload a file to generate a timeline JSON preview.")
