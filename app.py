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
# TITLE
# =============================
st.title("💼 Job Finding Prediction System")
st.markdown(
    """
    This application predicts whether a person is likely to **find a job**
    based on education, skills, experience, and digital readiness.
    """
)

st.divider()

# =============================
# USER INPUTS
# =============================
st.subheader("📊 Enter Candidate Information")

education = st.slider("🎓 Education Level", 0.0, 10.0, 5.0)
technical = st.slider("🛠 Technical Skills", 0.0, 10.0, 5.0)
soft = st.slider("🤝 Soft Skills", 0.0, 10.0, 5.0)
experience = st.slider("📄 Work Experience", 0.0, 10.0, 5.0)
job_search = st.slider("🔍 Job Search Activity", 0.0, 10.0, 5.0)
digital = st.slider("💻 Digital Presence", 0.0, 10.0, 5.0)

# =============================
# BUILD FEATURE VECTOR
# =============================
# Initialize all features with 0
input_data = pd.DataFrame(
    np.zeros((1, len(feature_names))),
    columns=feature_names
)

# Inject values into representative feature groups
for col in feature_names:
    if col.startswith("edu_"):
        input_data[col] = education
    elif col.startswith("tech_"):
        input_data[col] = technical
    elif col.startswith("soft_"):
        input_data[col] = soft
    elif col.startswith("work_"):
        input_data[col] = experience
    elif col.startswith("jobsearch_"):
        input_data[col] = job_search
    elif col.startswith("digital_"):
        input_data[col] = digital

# =============================
# PREDICTION
# =============================
st.divider()

if st.button("🔮 Predict Job Outcome"):
    probability = model.predict_proba(input_data)[0][1]
    prediction = int(probability > 0.42)

    st.subheader("📌 Prediction Result")

    if prediction == 1:
        st.success(f"✅ Likely to get a job\n\nConfidence: {probability:.2%}")
    else:
        st.error(f"❌ Unlikely to get a job\n\nConfidence: {1 - probability:.2%}")

    st.progress(probability)

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption("AI-based Job Prediction System | MSc AI Project")
