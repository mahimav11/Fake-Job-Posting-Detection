import streamlit as st
import pandas as pd
import joblib
import os


# ================================
# Page setup
# ================================
st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🛡️", layout="wide")
st.title("🛡️ Fake Job Posting Detector")
st.write("Protect yourself from fraudulent job postings. Enter details below:")


st.write("Current directory:", os.getcwd())
st.write("Files in model folder:", os.listdir("model"))

# ================================
# Load model safely
# ================================
model_path = "model/fraud_job_pipeline.pkl"
model = None

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.error(f"Model file not found at path: {model_path}")

# ================================
# Input form
# ================================
with st.form("job_form"):
    title = st.text_input("Job Title", "Software Engineer")
    location = st.text_input("Location", "New York")
    department = st.text_input("Department", "Engineering")
    salary_range = st.text_input("Salary Range", "$80,000 - $100,000")
    employment_type = st.selectbox("Employment Type", ["", "Full-time", "Part-time", "Contract", "Temporary", "Internship"])
    required_experience = st.text_input("Required Experience", "Mid-Senior level")
    required_education = st.text_input("Required Education", "Bachelors")
    industry = st.text_input("Industry", "Information Technology")
    function = st.text_input("Function", "Software Development")
    description = st.text_area("Job Description", "Looking for an experienced software engineer...")
    requirements = st.text_area("Requirements", "Must know Python, Java, SQL.")
    benefits = st.text_area("Benefits", "Health insurance, flexible hours.")
    telecommuting = st.selectbox("Telecommuting", [0, 1])
    has_company_logo = st.selectbox("Has Company Logo", [0, 1])
    has_questions = st.selectbox("Has Questions", [0, 1])
    company_profile = st.text_area("Company Profile", "Leading IT company providing software solutions.")
    
    submitted = st.form_submit_button("🔍 Analyze")

# ================================
# Prediction
# ================================
if submitted:
    if model:
        input_data = {
            "title": title,
            "location": location,
            "department": department,
            "salary_range": salary_range,
            "employment_type": employment_type,
            "required_experience": required_experience,
            "required_education": required_education,
            "industry": industry,
            "function": function,
            "description": description,
            "requirements": requirements,
            "benefits": benefits,
            "telecommuting": int(telecommuting),
            "has_company_logo": int(has_company_logo),
            "has_questions": int(has_questions),
            "company_profile": company_profile,
            "title_length": len(str(title).split()),
            "description_length": len(str(description).split()),
            "requirements_length": len(str(requirements).split())
        }

        df = pd.DataFrame([input_data])

        try:
            prob_fake = model.predict_proba(df)[:, 1][0]
            prediction = "🚨 Fake Job" if prob_fake >= 0.5 else "✅ Real Job"
            st.subheader("Result")
            st.metric("Prediction", prediction, f"{prob_fake*100:.2f}% confidence")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.error("Model is not loaded. Cannot make predictions.")
