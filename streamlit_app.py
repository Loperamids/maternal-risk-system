
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Maternal Risk Rapid Assessment System",
    page_icon=":hospital:",
    layout="wide"
)

# ========================= SESSION STATE =========================
if "patient_verified" not in st.session_state:
    st.session_state.patient_verified = False
if "verified_patient_name" not in st.session_state:
    st.session_state.verified_patient_name = ""
if "verified_patient_id" not in st.session_state:
    st.session_state.verified_patient_id = ""

# ========================= CSS =========================
st.markdown("""
    <style>
    .risk-high {color:red;font-weight:bold;}
    .risk-low {color:green;font-weight:bold;}
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

# ========================= LOAD MODEL =========================
@st.cache_resource
def load_model():
    if not os.path.exists("maternal_model.pkl"):
        st.error("Model not found. Run training first.")
        st.stop()

    model = joblib.load("maternal_model.pkl")
    scaler = joblib.load("scaler.pkl")

    yte = joblib.load("ytest.pkl")
    ypred = joblib.load("ypred.pkl")

    metrics = {
        "accuracy": accuracy_score(yte, ypred),
        "precision": precision_score(yte, ypred),
        "recall": recall_score(yte, ypred),
        "f1": f1_score(yte, ypred)
    }

    return model, scaler, metrics, yte, ypred

model, scaler, metrics, yte, ypred = load_model()

# ========================= TABS =========================
tab1, tab2 = st.tabs(["Risk Assessment", "My Records"])

# ========================= TAB 1 =========================
with tab1:

    if not st.session_state.patient_verified:
        st.subheader("Patient Verification")

        ptype = st.radio("Patient Type", ["New Patient", "Existing Patient"])

        if ptype == "New Patient":
            pname = st.text_input("Patient Name")

            if st.button("Register"):
                if pname:
                    st.session_state.verified_patient_name = pname
                    st.session_state.verified_patient_id = generate_patient_id()
                    st.session_state.patient_verified = True
                    st.success(f"Assigned ID: {st.session_state.verified_patient_id}")
                    st.rerun()
                else:
                    st.error("Enter patient name")

        else:
            pname = st.text_input("Patient Name")
            pid = st.text_input("Patient ID")

            if st.button("Verify"):
                if os.path.exists("patient_records.csv"):
                    df = pd.read_csv("patient_records.csv")
                    match = df[
                        (df["Patient_ID"] == pid) &
                        (df["Patient_Name"].str.lower() == pname.lower())
                    ]
                    if not match.empty:
                        st.session_state.patient_verified = True
                        st.session_state.verified_patient_name = pname
                        st.session_state.verified_patient_id = pid
                        st.success("Verified")
                        st.rerun()
                    else:
                        st.error("Not found")

        st.stop()

    # ========================= INPUT =========================
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

        # Ensure correct order
        input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

        input_scaled = scaler.transform(input_df)

        try:
            pred = model.predict(input_scaled)[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        risk = "High Risk" if pred == 1 else "Low Risk"

        if pred == 1:
            st.error(f"Predicted Risk: {risk}")
        else:
            st.success(f"Predicted Risk: {risk}")

        # ================= METRICS =================
        st.subheader("Model Performance")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        col2.metric("Precision", f"{metrics['precision']:.3f}")
        col3.metric("Recall", f"{metrics['recall']:.3f}")
        col4.metric("F1 Score", f"{metrics['f1']:.3f}")

        # ================= SAVE RECORD =================
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

    if st.button("Finish"):
        st.session_state.patient_verified = False
        st.rerun()

# ========================= TAB 2 =========================
with tab2:
    st.subheader("My Records")

    pid = st.text_input("Patient ID")
    pname = st.text_input("Patient Name")

    if st.button("Search"):
        if os.path.exists("patient_records.csv"):
            df = pd.read_csv("patient_records.csv")
            st.dataframe(df[
                (df["Patient_ID"] == pid)  &
                (df["Patient_Name"].str.lower() == pname.lower())
            ])
