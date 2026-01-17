# app.py — NephroBridge (Product-grade, Hackathon-ready)

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

st.set_page_config(
    page_title="NephroBridge",
    page_icon="🧠",
    layout="centered"
)

# --------------------------------------------------
# HEADER (Product framing)
# --------------------------------------------------
st.title("NephroBridge")
st.subheader("Clear explanations for kidney labs, medications, and next steps")

st.markdown(
    """
NephroBridge uses **Google’s MedGemma (HAI-DEF)** models to explain
complex kidney-related information in **calm, patient-friendly language**.

Built for moments when lab results change and answers feel unclear.
"""
)

st.divider()

# --------------------------------------------------
# STEP 1 — Journey stage + model toggle
# --------------------------------------------------
st.markdown("## Step 1 — Your kidney journey")

stage = st.radio(
    "Where are you right now?",
    {
        "Post-transplant recovery": "post_transplant",
        "Advanced CKD (not on dialysis)": "advanced_ckd",
        "On dialysis": "dialysis",
    },
    format_func=lambda x: x,
)

st.markdown("## Step 1a — AI model")

model_choice = st.radio(
    "Which AI model should explain your information?",
    [
        "MedGemma (Healthcare-specialized, HAI-DEF)",
        "Gemma-2B (Lightweight, general-purpose)",
    ],
)

USE_MEDGEMMA = model_choice.startswith("MedGemma")
os.environ["USE_MEDGEMMA"] = "true" if USE_MEDGEMMA else "false"

st.caption(
    "MedGemma is trained specifically on healthcare data and is recommended "
    "for patient-facing explanations."
)

st.divider()

# --------------------------------------------------
# STEP 2 — Patient-friendly data input
# --------------------------------------------------
st.markdown("## Step 2 — Add your recent information")

input_mode = st.radio(
    "How would you like to add your information?",
    ["Fill a simple form", "Upload a file"],
)

timeline = None
error = None

def build_timeline(stage_key, reason, labs):
    return {
        "kidney_journey_stage": stage_key,
        "recent_hospital_stay": (
            {"reason_for_admission": reason} if reason else {}
        ),
        "labs_over_time": labs,
        "medications_before": [],
        "medications_after": [],
        "followup_appointments": [],
        "pending_labs": [],
    }

if input_mode == "Fill a simple form":
    with st.form("patient_form"):
        reason = st.text_input(
            "Why were you seen recently? (optional)",
            placeholder="Routine transplant follow-up, dehydration, lab review",
        )

        st.markdown("### Lab results")
        st.caption("Example: Creatinine 1.6 mg/dL on 2026-01-08")

        lab_rows = st.data_editor(
            [{"date": "", "lab_name": "", "value": "", "unit": "", "reference_range": ""}],
            num_rows="dynamic",
            use_container_width=True,
        )

        submitted = st.form_submit_button("Continue")

    if submitted:
        labs = [
            r for r in lab_rows
            if r.get("lab_name") and r.get("value")
        ]

        if not labs:
            error = "Please add at least one lab result."
        else:
            timeline = build_timeline(stage, reason, labs)

else:
    uploaded = st.file_uploader("Upload a CSV or JSON file", type=["csv", "json"])

    if uploaded:
        try:
            if uploaded.name.endswith(".json"):
                timeline = json.load(uploaded)
                timeline["kidney_journey_stage"] = stage
            else:
                df = pd.read_csv(uploaded)
                labs = []
                for _, r in df.iterrows():
                    labs.append({
                        "date": str(r.get("date", "")),
                        "lab_name": str(r.get("lab_name", "")),
                        "value": str(r.get("value", "")),
                        "unit": str(r.get("unit", "")),
                        "reference_range": str(r.get("reference_range", "")),
                    })
                timeline = build_timeline(stage, "", labs)
        except Exception as e:
            error = str(e)

if error:
    st.error(error)

# --------------------------------------------------
# STEP 3 — Primary action
# --------------------------------------------------
if timeline:
    st.divider()
    st.markdown("## Step 3 — Understand what’s going on")

    if st.button("Explain my situation", use_container_width=True):
        with st.spinner("Analyzing your information with AI…"):
            lab_text = run_lab_agent_from_timeline(timeline)
            med_text = run_medication_agent(timeline)
            follow_text = run_followup_agent(timeline)

        st.divider()

        # --------------------------------------------------
        # RESULTS — Tabbed, clean, patient-first
        # --------------------------------------------------
        tab1, tab2, tab3 = st.tabs(
            ["🧪 Lab changes", "💊 Medications", "🧾 What’s next"]
        )

        with tab1:
            st.markdown(lab_text)

        with tab2:
            st.markdown(med_text)

        with tab3:
            st.markdown(follow_text)

        # Optional transparency (collapsed)
        with st.expander("See the structured timeline used by the AI"):
            st.json(timeline)

else:
    st.info("Add your information above to continue.")
