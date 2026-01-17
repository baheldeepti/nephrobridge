# app.py — NephroBridge Hackathon Demo
import json
import os
from pathlib import Path
import streamlit as st
import pandas as pd

from lab_agent_v1 import run_lab_agent_from_timeline
from medication_agent import run_medication_agent
from followup_agent import run_followup_agent

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "DATA"
DATA_DIR.mkdir(exist_ok=True)

USE_MEDGEMMA = os.getenv("USE_MEDGEMMA", "false").lower() == "true"

st.set_page_config(
    page_title="NephroBridge",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("NephroBridge")
st.caption(
    "A human-centered AI assistant using Google MedGemma "
    "to explain kidney labs, medications, and follow-ups."
)

st.markdown(
    f"**Model in use:** {'MedGemma (HAI-DEF)' if USE_MEDGEMMA else 'Gemma-2B (local dev)'}"
)

# --------------------------------------------------
# STAGE SELECTION
# --------------------------------------------------
st.markdown("## 1️⃣ Where are you in your kidney journey?")
stage = st.radio(
    "Choose one:",
    ["post_transplant", "advanced_ckd", "dialysis"],
    horizontal=True
)

# --------------------------------------------------
# DATA INPUT
# --------------------------------------------------
st.markdown("## 2️⃣ Add your information")
mode = st.radio(
    "How would you like to provide your info?",
    ["Fill a simple form", "Upload a CSV or JSON file"],
    horizontal=True
)

timeline = None
error = None

def normalize_labs(df):
    labs = []
    for _, r in df.iterrows():
        labs.append({
            "date": str(r.get("date", "")),
            "lab_name": str(r.get("lab_name", "")),
            "value": str(r.get("value", "")),
            "unit": str(r.get("unit", "")),
            "reference_range": str(r.get("reference_range", "")),
        })
    return labs

if mode == "Fill a simple form":
    with st.form("patient_form"):
        reason = st.text_input(
            "Why were you recently seen in clinic or hospital? (optional)",
            placeholder="Routine transplant follow-up, dehydration, lab review"
        )

        st.markdown("### Labs")
        labs_table = st.data_editor(
            [{"date": "", "lab_name": "", "value": "", "unit": "", "reference_range": ""}],
            num_rows="dynamic",
            use_container_width=True
        )

        submitted = st.form_submit_button("Create timeline")

    if submitted:
        labs = [r for r in labs_table if r["lab_name"] and r["value"]]
        if not labs:
            error = "Please add at least one lab result."
        else:
            timeline = {
                "kidney_journey_stage": stage,
                "recent_hospital_stay": {"reason_for_admission": reason} if reason else {},
                "labs_over_time": labs,
                "medications_before": [],
                "medications_after": [],
                "followup_appointments": [],
                "pending_labs": [],
            }

else:
    uploaded = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])
    if uploaded:
        try:
            if uploaded.name.endswith(".json"):
                timeline = json.load(uploaded)
                timeline["kidney_journey_stage"] = stage
            else:
                df = pd.read_csv(uploaded)
                timeline = {
                    "kidney_journey_stage": stage,
                    "labs_over_time": normalize_labs(df),
                }
        except Exception as e:
            error = str(e)

# --------------------------------------------------
# PREVIEW
# --------------------------------------------------
if error:
    st.error(error)

if timeline:
    st.success("Timeline created")
    st.json(timeline)

    st.divider()

    # --------------------------------------------------
    # AGENT ORCHESTRATION
    # --------------------------------------------------
    st.markdown("## 3️⃣ AI explanations (powered by MedGemma)")

    if st.button("Explain my situation"):
        with st.spinner("Running AI agents…"):
            lab_out = run_lab_agent_from_timeline(timeline)
            med_out = run_medication_agent(timeline)
            follow_out = run_followup_agent(timeline)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🧪 Lab Explanation")
            st.write(lab_out)

        with col2:
            st.markdown("### 💊 Medication Changes")
            st.write(med_out)

        with col3:
            st.markdown("### 🧾 Follow-ups & Open Loops")
            st.write(follow_out)
else:
    st.info("Add your information to begin.")
