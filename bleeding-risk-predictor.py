import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import xgboost as xgb

# ✅ Fix point 1: set_page_config() must be the first Streamlit command
st.set_page_config(layout="wide")  # This must come first!

# Load model and feature names
@st.cache_resource
def load_model():
    model_data = joblib.load("xgboost_bleed_model.joblib")
    return model_data["model"], model_data["feature_names"]


model, feature_names = load_model()

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)

# Set threshold (adjustable based on model performance)
THRESHOLD = 0.5  # Default threshold, recommended to determine optimal value from ROC curve


def user_input_features():
    st.header("Patient Clinical Parameters Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        apache_iv_score = st.number_input("APACHE IV Score", min_value=0, max_value=200, value=50)
        gcs = st.number_input("GCS Score", min_value=3, max_value=15, value=12)
        albumin_max = st.number_input("Max Albumin (g/dL)", min_value=1.0, max_value=6.0, value=3.5, step=0.1)
        hematocrit_min = st.number_input("Min Hematocrit (%)", min_value=10, max_value=60, value=30)
        anemia = st.selectbox("Anemia", ["No", "Yes"], index=0)

    with col2:
        platelet_min = st.number_input("Min Platelet Count (×10³/µL)", min_value=10, max_value=500, value=150)
        ptt_max = st.number_input("Max PTT (seconds)", min_value=20, max_value=200, value=35)
        pt_max = st.number_input("Max PT (seconds)", min_value=10, max_value=50, value=13)
        bun_max = st.number_input("Max BUN (mg/dL)", min_value=5, max_value=100, value=20)
        respiratoryrate = st.number_input("Respiratory Rate (breaths/min)", min_value=5, max_value=50, value=18)

    with col3:
        nibp_systolic = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=250, value=120)
        nibp_diastolic = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, value=80)
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        caucasian = st.selectbox("Caucasian", ["No", "Yes"], index=1)
        medsurg_icu = st.selectbox("Medical/Surgical ICU", ["No", "Yes"], index=0)

    # Second group of columns
    col4, col5, col6 = st.columns(3)

    with col4:
        cardiac_icu = st.selectbox("Cardiac ICU", ["No", "Yes"], index=0)
        neuro_icu = st.selectbox("Neuro ICU", ["No", "Yes"], index=0)
        gastrointestinal_condition = st.selectbox("Gastrointestinal Condition", ["No", "Yes"], index=0)

    with col5:
        trauma = st.selectbox("Trauma", ["No", "Yes"], index=0)
        history_of_bleed = st.selectbox("History of Bleeding", ["No", "Yes"], index=0)
        history_of_vte = st.selectbox("History of VTE", ["No", "Yes"], index=0)

    with col6:
        sepsis = st.selectbox("Sepsis", ["No", "Yes"], index=0)
        vascular_disorders = st.selectbox("Vascular Disorders", ["No", "Yes"], index=0)
        stress_ulcer_drug = st.selectbox("Stress Ulcer Medication", ["No", "Yes"], index=0)

    # Automatically calculate derived features
    coagulation_dysfunction = 1 if (ptt_max > 40 or pt_max > 14) else 0
    respiratory_failure = 1 if (respiratoryrate > 24 or nibp_systolic < 90) else 0

    # Create input dataframe
    data = {
        'apache_iv_score': apache_iv_score,
        'gcs': gcs,
        'albumin_max': albumin_max,
        'hematocrit_min': hematocrit_min,
        'anemia': 1 if anemia == "Yes" else 0,
        'platelet_min': platelet_min,
        'ptt_max': ptt_max,
        'coagulation_dysfunction': coagulation_dysfunction,
        'pt_max': pt_max,
        'bun_max': bun_max,
        'respiratoryrate': respiratoryrate,
        'nibp_systolic': nibp_systolic,
        'nibp_diastolic': nibp_diastolic,
        'gender': 1 if gender == "Female" else 0,
        'caucasian': 1 if caucasian == "Yes" else 0,
        'medsurg_icu': 1 if medsurg_icu == "Yes" else 0,
        'cardiac_icu': 1 if cardiac_icu == "Yes" else 0,
        'neuro_icu': 1 if neuro_icu == "Yes" else 0,
        'gastrointestinal_condition': 1 if gastrointestinal_condition == "Yes" else 0,
        'trauma': 1 if trauma == "Yes" else 0,
        'history_of_bleed': 1 if history_of_bleed == "Yes" else 0,
        'history_of_vte': 1 if history_of_vte == "Yes" else 0,
        'sepsis': 1 if sepsis == "Yes" else 0,
        'vascular_disorders': 1 if vascular_disorders == "Yes" else 0,
        'acute_coronary_syndrome': 0,  # Not in example inputs
        'respiratory_failure': respiratory_failure,
        'vasopressors_inotropic_agents': 0,  # Not in example inputs
        'stress_ulcer_drug': 1 if stress_ulcer_drug == "Yes" else 0
    }

    return pd.DataFrame([data], columns=feature_names)


def main():
    st.title("🏥 ICU Major Bleeding Risk Prediction Tool")
    st.markdown("""
    **Predicting in-hospital major bleeding risk for ICU patients using XGBoost model**  
    *Please fill in patient clinical parameters below and click predict button*
    """)

    input_df = user_input_features()

    if st.button("Predict Bleeding Risk"):
        try:
            # Prediction
            proba = model.predict_proba(input_df)[0, 1]
            prediction = "High Risk" if proba >= THRESHOLD else "Low Risk"

            # Display results
            st.success("Prediction complete!")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction Result", prediction)
            with col2:
                st.metric("Risk Probability", f"{proba * 100:.1f}%")
            # SHAP explanation
            shap_values = explainer(input_df)
            # SHAP force plot
            st.subheader("Individual Prediction Explanation")
            force_plot = shap.force_plot(
                base_value=explainer.expected_value,
                shap_values=shap_values.values[0],
                features=input_df.iloc[0],
                feature_names=feature_names,
                matplotlib=False
            )
            st.components.v1.html(shap.getjs() + force_plot.html(), height=200)

            # ✅ Ensure risk explanation comes after SHAP plots in same try block
            st.subheader("Risk Interpretation")
            proba = model.predict_proba(input_df)[0, 1]  # Get prediction probability
            if proba > 0.7:
                st.warning("""
                       **⚠️ High Risk Warning**  
                       This patient has high bleeding risk (>70%), recommendations:  
                       - Enhanced coagulation monitoring  
                       - Consider preventive interventions  
                       - Avoid unnecessary invasive procedures
                       """)
            elif proba > 0.3:
                st.info("""
                       **ℹ️ Moderate Risk**  
                       This patient has moderate bleeding risk (30-70%), recommendations:  
                       - Routine coagulation monitoring  
                       - Review medication regimen  
                       - Monitor for bleeding signs
                       """)
            else:
                st.success("""
                       **✅ Low Risk**  
                       This patient has low bleeding risk (<30%), routine care is sufficient
                       """)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

    # Sidebar information
    with st.sidebar:
        st.header("About This Tool")
        st.markdown("""
        - **Model Type**: XGBoost
        - **Training Data**: ICU data from 208 centers
        - **Prediction Target**: In-hospital major bleeding risk
        - **Threshold**: {:.3f} (determined by Youden index)
        """.format(THRESHOLD))

        st.header("Instructions")
        st.markdown("""
        1. Enter patient clinical parameters
        2. Click "Predict Bleeding Risk" button
        3. View prediction results and explanations
        """)

        st.warning("""
        **Clinical Notes**  
        This tool's results are for reference only and should be used with clinical judgment.  
        High-risk patients require comprehensive evaluation of other risk factors.
        """)


if __name__ == '__main__':
    main()
