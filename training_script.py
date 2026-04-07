
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer

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

# ========================= ADD NOISE (OPTIONAL) =========================
for col in ["systolic_bp", "diastolic_bp", "blood_sugar", "temperature"]:
    df[col] = df[col] + np.random.normal(0, 0.5, size=len(df))

# ========================= SPLIT FEATURES & TARGET =========================
X = df.drop("risk", axis=1)
y = df["risk"]

# ========================= SCALE FEATURES =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========================= MODEL =========================
model = RandomForestClassifier(
    n_estimators=120,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
)

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}

# ========================= TRAIN FINAL MODEL =========================
Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
model.fit(Xtr, ytr)

# ========================= TRAIN & TEST METRICS =========================
ytrain_pred = model.predict(Xtr)
ytest_pred = model.predict(Xte)

train_metrics = {
    'accuracy': accuracy_score(ytr, ytrain_pred),
    'precision': precision_score(ytr, ytrain_pred),
    'recall': recall_score(ytr, ytrain_pred),
    'f1': f1_score(ytr, ytrain_pred)
}

test_metrics = {
    'accuracy': accuracy_score(yte, ytest_pred),
    'precision': precision_score(yte, ytest_pred),
    'recall': recall_score(yte, ytest_pred),
    'f1': f1_score(yte, ytest_pred)
}

# ========================= SAVE MODEL & ARTIFACTS =========================
joblib.dump(model, 'maternal_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(train_metrics, 'train_metrics.pkl')
joblib.dump(test_metrics, 'metrics.pkl')
joblib.dump(yte, 'ytest.pkl')
joblib.dump(ytest_pred, 'ypred.pkl')

print('\\nFinal Model Trained and Saved ✓')
print(f'Train metrics: {train_metrics}')
print(f'Test metrics: {test_metrics}')

