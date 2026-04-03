
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ========================= LOAD DATASETS =========================
df1 = pd.read_excel("maternal_cleaned_bp-1.xlsx", engine='openpyxl')
df2 = pd.read_excel("final_processed_dataset.xlsx", engine='openpyxl')

# ========================= COMBINE DATASETS =========================
df = pd.concat([df1, df2], ignore_index=True)

# ========================= CLEAN COLUMN NAMES =========================
df.columns = df.columns.str.strip().str.lower()
df = df.rename(columns={
    "body_temperature": "temperature",
    "systolic": "systolic_bp",
    "diastolic": "diastolic_bp",
    "pre-pregnancy_weight": "pre_pregnancy_weight",
    "risk_level": "risk"
})
df["risk"] = df["risk"].map({"Low": 0, "High": 1, "Low Risk": 0, "High Risk": 1})
df = df.drop(columns=["id", "name"], errors="ignore")
df = df[["age","systolic_bp","diastolic_bp","blood_sugar","temperature","heart_rate",
         "maternal_weight","pre_pregnancy_weight","fetal_age","risk"]]
df.dropna(subset=["risk"], inplace=True)

# ========================= ADD NOISE (ANTI-OVERFITTING) =========================
for col in ["systolic_bp", "diastolic_bp", "blood_sugar", "temperature"]:
    df[col] = df[col] + np.random.normal(0, 0.5, size=len(df))

# ========================= SPLIT FEATURES & TARGET =========================
X = df.drop("risk", axis=1)
y = df["risk"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ========================= SCALE FEATURES =========================
scaler = StandardScaler()
Xtr_scaled = scaler.fit_transform(Xtr)
Xte_scaled = scaler.transform(Xte)

# ========================= TRAIN MODEL =========================
model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
model.fit(Xtr_scaled, ytr)

# ========================= TRAIN METRICS =========================
ytrain_pred = model.predict(Xtr_scaled)

train_metrics = {
    "accuracy": accuracy_score(ytr, ytrain_pred),
    "precision": precision_score(ytr, ytrain_pred),
    "recall": recall_score(ytr, ytrain_pred),
    "f1": f1_score(ytr, ytrain_pred)
}
# ========================= EVALUATE MODEL =========================
ypred = model.predict(Xte_scaled)
metrics = {
    "accuracy": accuracy_score(yte, ypred),
    "precision": precision_score(yte, ypred),
    "recall": recall_score(yte, ypred),
    "f1": f1_score(yte, ypred)
}

# ========================= SAVE MODEL & ARTIFACTS =========================
joblib.dump(model, "maternal_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(metrics, "metrics.pkl")
joblib.dump(yte, "ytest.pkl")
joblib.dump(ypred, "ypred.pkl")
print("Model trained and saved ✓")
print(f"Test metrics: {metrics}")
