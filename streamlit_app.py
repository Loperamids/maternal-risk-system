

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Maternal Risk Rapid Assessment System",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="expanded"
)

if st.button("Clear Cache"):
    st.cache_resource.clear()

# ========================= INITIALIZE SESSION STATE =========================
if "patient_verified" not in st.session_state:
    st.session_state.patient_verified = False
if "verified_patient_name" not in st.session_state:
    st.session_state.verified_patient_name = ""
if "verified_patient_id" not in st.session_state:
    st.session_state.verified_patient_id = ""

# ========================= CSS =========================
st.markdown("""
    <style>
    .main-header {font-size:2.5em;font-weight:bold;color:#2E8B57;text-align:center;}
    .sub-header {font-size:1.5em;font-weight:bold;color:#4682B4;margin-top:20px;}
    .risk-high {color:red;font-weight:bold;}
    .risk-low {color:green;font-weight:bold;}
    .info-box {background:#e7f3fe;border-left:6px solid #2196F3;padding:10px;}
    </style>
""", unsafe_allow_html=True)

# ========================= PATIENT ID =========================
def generate_patient_id():
    file = "patient_records.csv"
    if os.path.exists(file):
        df = pd.read_csv(file)
        if not df.empty:
            nums = df["Patient_ID"].str.extract('P(\d+)').dropna().astype(int)
            return f"P{nums.max().values[0]+1:04d}"
    return "P0001"

# ========================= MODEL =========================
@st.cache_resource
def load_model():
    if os.path.exists("maternal_model.pkl") and os.path.exists("scaler.pkl"):
        model = joblib.load("maternal_model.pkl")
        scaler = joblib.load("scaler.pkl")

        # Load test data
        if os.path.exists("ytest.pkl") and os.path.exists("ypred.pkl"):
            yte = joblib.load("ytest.pkl")
            ypred = joblib.load("ypred.pkl")
        else:
            st.error("Missing ytest.pkl or ypred.pkl")
            st.stop()

        # RECOMPUTE METRICS (THIS IS THE FIX)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        metrics = {
            "accuracy": accuracy_score(yte, ypred),
            "precision": precision_score(yte, ypred),
            "recall": recall_score(yte, ypred),
            "f1": f1_score(yte, ypred)
        }

        return model, scaler, metrics, yte, ypred

    else:
        st.error("Model files not found. Please run the training script first.")
        st.stop()

# ===== LOAD MODEL =====
model, scaler, metrics, yte, ypred = load_model()

# ========================= TABS =========================
tab1, tab2 = st.tabs(["Risk Assessment", "My Records"])

# ========================= TAB 1 =========================
with tab1:

    if not st.session_state.patient_verified:
        st.subheader("Patient Verification")

        ptype = st.radio("Patient Type", ["New Patient", "Existing Patient"])

        if ptype == "New Patient":
            pname = st.text_input("Patient Name", key="new_patient_name")

            if st.button("Register New Patient"):
                if pname:
                    st.session_state.verified_patient_name = pname
                    st.session_state.verified_patient_id = generate_patient_id()
                    st.session_state.patient_verified = True
                    st.success(f"Assigned ID: {st.session_state.verified_patient_id}")
                    st.rerun()
                else:
                    st.error("Enter patient name")

        else:
            pname = st.text_input("Patient Name", key="verify_name")
            pid = st.text_input("Patient ID", key="verify_id")

            if st.button("Proceed to Assessment"):
                if os.path.exists("patient_records.csv"):
                    df = pd.read_csv("patient_records.csv")

                    match = df[
                        (df["Patient_ID"] == pid) &
                        (df["Patient_Name"].str.lower() == pname.lower())
                    ]

                    if not match.empty:
                        st.session_state.verified_patient_name = pname
                        st.session_state.verified_patient_id = pid
                        st.session_state.patient_verified = True
                        st.success("Patient verified")
                        st.rerun()
                    else:
                        st.error("Patient not found")
                else:
                    st.error("No patient records file found")

        st.stop()

    st.subheader("Input Patient Data")

    patient_id = st.session_state.verified_patient_id
    patient_name = st.session_state.verified_patient_name

    st.text_input("Patient ID", patient_id, disabled=True)
    st.text_input("Patient Name", patient_name, disabled=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", value=25)
        sbp = st.number_input("Systolic BP", value=115)
        dbp = st.number_input("Diastolic BP", value=75)

    with col2:
        sugar = st.number_input("Blood Sugar", value=95.0)
        temp = st.number_input("Temperature", value=37.0)
        hr = st.number_input("Heart Rate", value=78)

    with col3:
        mw = st.number_input("Maternal Weight", value=65.0)
        pw = st.number_input("Pre-Pregnancy Weight", value=60.0)
        fa = st.number_input("Fetal Age", value=28)

    if st.button("Assess Risk"):

        input_df = pd.DataFrame([{
            "age": age,
            "systolic_bp": sbp,
            "diastolic_bp": dbp,
            "blood_sugar": sugar,
            "temperature": temp,
            "heart_rate": hr,
            "maternal_weight": mw,
            "pre_pregnancy_weight": pw,
            "fetal_age": fa
        }])

        try:
            input_df = input_df[scaler.feature_names_in_]
        except:
            pass

        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]

        risk = "High Risk" if pred == 1 else "Low Risk"

        st.error(f"Predicted Risk: {risk}") if risk == "High Risk" else st.success(f"Predicted Risk: {risk}")

        record = pd.DataFrame([{
            "Patient_ID": patient_id,
            "Patient_Name": patient_name,
            "Age": age,
            "Systolic_BP": sbp,
            "Diastolic_BP": dbp,
            "Blood_Sugar": sugar,
            "Temperature": temp,
            "Heart_Rate": hr,
            "Maternal_Weight": mw,
            "Pre_Pregnancy_Weight": pw,
            "Fetal_Age": fa,
            "Risk": risk,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])

        if os.path.exists("patient_records.csv"):
            record = pd.concat([pd.read_csv("patient_records.csv"), record], ignore_index=True)

        record.to_csv("patient_records.csv", index=False)

    if st.button("Finish & New Patient"):
        st.session_state.patient_verified = False
        st.rerun()

# ========================= TAB 2 =========================
with tab2:
    st.subheader("My Records")

    ppid = st.text_input("Patient ID", key="search_pid")
    pname = st.text_input("Patient Name", key="search_name")

    if st.button("Search My Records"):
        if os.path.exists("patient_records.csv"):
            df = pd.read_csv("patient_records.csv")
            st.dataframe(df[
                (df["Patient_ID"] == ppid) &
                (df["Patient_Name"].str.lower() == pname.lower())
            ])
