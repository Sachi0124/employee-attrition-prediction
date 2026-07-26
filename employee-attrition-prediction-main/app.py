import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Employee Attrition Risk Analyzer",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern Custom Styling (Obsidian, Charcoal & Emerald)
st.markdown("""
<style>
    .stApp { background-color: #09090b; color: #e4e4e7; font-family: -apple-system, sans-serif; }
    .app-title { font-size: 2rem; font-weight: 700; color: #f4f4f5; margin-bottom: 0.25rem; }
    .app-subtitle { color: #71717a; font-size: 0.95rem; margin-bottom: 2rem; }
    label { color: #a1a1aa !important; font-weight: 500 !important; font-size: 0.88rem !important; }
    div.stButton > button:first-child {
        background-color: #059669 !important; color: #ffffff !important;
        border: 1px solid #10b981 !important; padding: 0.6rem 1.5rem !important;
        border-radius: 8px !important; font-weight: 600 !important; width: 100% !important;
    }
    div.stButton > button:first-child:hover { background-color: #10b981 !important; }
    .result-stay { background-color: #064e3b; border: 1px solid #10b981; color: #a7f3d0; padding: 1.25rem; border-radius: 8px; text-align: center; }
    .result-leave { background-color: #7f1d1d; border: 1px solid #ef4444; color: #fecaca; padding: 1.25rem; border-radius: 8px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# 3. Load Trained Model & Scaler safely
@st.cache_resource
def load_artifacts():
    model, scaler = None, None
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
    if os.path.exists('scaler.pkl'):
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
    return model, scaler

model, scaler = load_artifacts()

# 4. App Header
st.markdown('<div class="app-title">Employee Attrition Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Predict employee retention risk based on workplace metrics and key performance indicators.</div>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ `model.pkl` not found in the project root directory. Please add it to continue.")
    st.stop()

# 5. Form Layout
st.subheader("📋 Employee Attributes")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Personal Details**")
    age = st.slider("Age", 18, 65, 32)
    distance = st.slider("Distance From Home (km)", 1, 50, 25) # Default to a higher risk distance
    num_companies = st.number_input("Prior Companies Worked", 0, 10, 5)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

with col2:
    st.markdown("**Job Profile**")
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    job_role = st.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician", 
        "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative"
    ])
    monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000, value=2500, step=500) # Lower income default
    overtime = st.selectbox("Overtime Required?", ["Yes", "No"])

with col3:
    st.markdown("**Satisfaction & Tenure**")
    env_satisfaction = st.select_slider("Environment Satisfaction", options=[1, 2, 3, 4], value=1)
    job_satisfaction = st.select_slider("Job Satisfaction", options=[1, 2, 3, 4], value=1)
    work_life_balance = st.select_slider("Work-Life Balance Rating", options=[1, 2, 3, 4], value=1)
    years_at_company = st.number_input("Years at Company", 0, 40, 2)

st.markdown("---")

# 6. Prediction Logic
if st.button("Evaluate Attrition Risk"):
    
    # We provide realistic "dataset average" defaults for the columns not in the UI
    # so the scaler doesn't crash on impossible zeros.
    baseline_employee = {
        'Age': age,
        'DistanceFromHome': distance,
        'MonthlyIncome': monthly_income,
        'NumCompaniesWorked': num_companies,
        'EnvironmentSatisfaction': env_satisfaction,
        'JobSatisfaction': job_satisfaction,
        'WorkLifeBalance': work_life_balance,
        'YearsAtCompany': years_at_company,
        'OverTime': overtime, # Passed as text so pd.get_dummies correctly creates OverTime_Yes
        'Department': department,
        'JobRole': job_role,
        'MaritalStatus': marital_status,
        
        # --- Background Baseline Attributes ---
        'DailyRate': 802,
        'Education': 3,
        'HourlyRate': 65,
        'JobInvolvement': 3,
        'JobLevel': 2,
        'MonthlyRate': 14313,
        'PercentSalaryHike': 15,
        'PerformanceRating': 3,
        'RelationshipSatisfaction': 3,
        'StockOptionLevel': 0 if marital_status == "Single" else 1,
        'TotalWorkingYears': max(years_at_company, age - 22), 
        'TrainingTimesLastYear': 3,
        'YearsInCurrentRole': max(0, years_at_company - 1),
        'YearsSinceLastPromotion': max(0, years_at_company - 2),
        'YearsWithCurrManager': max(0, years_at_company - 1),
        'BusinessTravel': 'Travel_Rarely',
        'EducationField': 'Life Sciences',
        'Gender': 'Male'
    }

    input_df = pd.DataFrame([baseline_employee])
    input_encoded = pd.get_dummies(input_df)

    try:
        expected_columns = model.feature_names_in_
        
        # Align features: missing categorical dummies (like Gender_Female) safely become 0
        final_input = input_encoded.reindex(columns=expected_columns, fill_value=0)

        # Scale data
        if scaler is not None:
            final_input = scaler.transform(final_input)

        # Predict
        prediction = model.predict(final_input)[0]
        probability = model.predict_proba(final_input)[0][1] if hasattr(model, "predict_proba") else None

        # Display
        res_col1, res_col2 = st.columns([2, 1])

        with res_col1:
            # We check if probability is greater than a threshold, or if the model outputs a hard 1
            if prediction == 1 or (probability and probability > 0.40):
                st.markdown(f"""
                    <div class="result-leave">
                        <h3 style="margin:0;">⚠️ High Risk of Attrition</h3>
                        <p style="margin: 0.5rem 0 0 0;">This employee displays key indicators linked to departure.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-stay">
                        <h3 style="margin:0;">✅ Low Risk (Likely to Stay)</h3>
                        <p style="margin: 0.5rem 0 0 0;">This employee's metrics indicate a strong retention likelihood.</p>
                    </div>
                """, unsafe_allow_html=True)

        with res_col2:
            if probability is not None:
                st.markdown(f"""
                    <div style="text-align: center; background-color: #18181b; border: 1px solid #27272a; padding: 1rem; border-radius: 8px; height: 100%;">
                        <div style="color: #ffffff; font-size: 0.95rem; font-weight: 600; margin-bottom: 0.2rem;">Calculated Risk Probability</div>
                        <div style="color: #ffffff; font-size: 2.4rem; font-weight: 800;">{probability * 100:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error making prediction: {e}")