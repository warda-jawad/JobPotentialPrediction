import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =============================
# LOAD MODEL & FEATURES
# =============================
model = joblib.load("job_model.pkl")
feature_names = joblib.load("model_features.pkl")

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Job Success Predictor",
    page_icon="💼",
    layout="centered"
)

# =============================
# HEADER / TITLE
# =============================
st.markdown("## ✨ Welcome to the Job Predictor ✨")
st.title("💼 Job Finding Prediction System")

st.markdown(
    """
This AI-based application estimates the **probability of finding a job**
using education, skills, experience, and job-search behavior.

⚠️ *Predictions are probabilistic and meant for decision support — not guarantees.*
"""
)
st.divider()

# =============================
# LAYOUT: LEFT INFO COLUMN + MAIN INPUTS
# =============================
col1, col2 = st.columns([0.3, 0.7])

# ----- LEFT INFO / DECORATION -----
with col1:
    st.markdown("### 💡 Tips")
    st.markdown(
        "- Adjust sliders according to your real profile\n"
        "- Higher education & skills → higher chance\n"
        "- Predictions are **probabilistic**, not guarantees"
    )
    
    st.markdown("### 📘 About")
    st.markdown(
        "This tool uses:\n"
        "- A trained **ML model**\n"
        "- **Agent-based simulation** data\n"
        "- Streamlit interactive UI"
    )

    st.markdown("### 📊 Project Stats")
    st.metric("Agents Simulated", "1000")
    st.metric("Features Used", "301")
    st.metric("ML Models", "4")

# ----- RIGHT MAIN INPUTS -----
with col2:
    st.subheader("📊 Candidate Profile")

    education = st.slider("🎓 Education Level", 0.0, 1.0, 0.5)
    technical = st.slider("🛠 Technical Skills", 0.0, 1.0, 0.5)
    soft = st.slider("🤝 Soft Skills", 0.0, 1.0, 0.5)
    experience = st.slider("📄 Work Experience", 0.0, 1.0, 0.5)
    job_search = st.slider("🔍 Job Search Activity", 0.0, 1.0, 0.5)
    digital = st.slider("💻 Digital Presence", 0.0, 1.0, 0.5)

    # Weak profile warning
    if education < 0.2 and technical < 0.2 and soft < 0.2 and experience < 0.2:
        st.warning(
            "⚠️ Very limited profile strength detected. Prediction confidence may be unreliable."
        )

    # Build feature vector safely
    input_data = pd.DataFrame(
        np.zeros((1, len(feature_names))),
        columns=feature_names
    )

    # Feature groups
    edu_cols = [c for c in feature_names if c.startswith("edu_")]
    tech_cols = [c for c in feature_names if c.startswith("tech_")]
    soft_cols = [c for c in feature_names if c.startswith("soft_")]
    work_cols = [c for c in feature_names if c.startswith("work_")]
    job_cols = [c for c in feature_names if c.startswith("jobsearch_")]
    digital_cols = [c for c in feature_names if c.startswith("digital_")]

    for c in edu_cols:
        input_data[c] = education / max(len(edu_cols), 1)
    for c in tech_cols:
        input_data[c] = technical / max(len(tech_cols), 1)
    for c in soft_cols:
        input_data[c] = soft / max(len(soft_cols), 1)
    for c in work_cols:
        input_data[c] = experience / max(len(work_cols), 1)
    for c in job_cols:
        input_data[c] = job_search / max(len(job_cols), 1)
    for c in digital_cols:
        input_data[c] = digital / max(len(digital_cols), 1)

    st.divider()

    # ----- PREDICTION BUTTON -----
    if st.button("🔮 Predict Job Outcome"):
        raw_proba = model.predict_proba(input_data)[0][1]

        # Calibrate displayed confidence
        confidence = min(max(raw_proba, 0.05), 0.95)
        prediction = int(raw_proba >= 0.42)

        st.subheader("📌 Prediction Result")

        if prediction == 1:
            st.success(
                f"✅ **Likely to get a job**\n\n"
                f"Estimated confidence: **{confidence:.2%}**"
            )
        else:
            st.error(
                f"❌ **Unlikely to get a job**\n\n"
                f"Estimated confidence: **{1 - confidence:.2%}**"
            )

        st.progress(confidence)
        st.caption(
            "ℹ️ Confidence reflects statistical likelihood, not certainty."
        )

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption("AI-based Job Prediction System | MSc Artificial Intelligence Project")
