
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

# ========================= REMOVE DUPLICATES =========================
df = df.drop_duplicates()

# ========================= CLEAN COLUMN NAMES =========================
df.columns = df.columns.str.strip().str.lower()
df = df.rename(columns={
    "body_temperature": "temperature",
    "systolic": "systolic_bp",
    "diastolic": "diastolic_bp",
    "pre-pregnancy_weight": "pre_pregnancy_weight",
    "risk_level": "risk"
})

# ========================= TARGET CLEANING =========================
df["risk"] = df["risk"].map({"Low": 0, "High": 1, "Low Risk": 0, "High Risk": 1})

# ========================= SELECT FEATURES =========================
df = df.drop(columns=["id", "name"], errors="ignore")
df = df[[
    "age","systolic_bp","diastolic_bp","blood_sugar","temperature","heart_rate",
    "maternal_weight","pre_pregnancy_weight","fetal_age","risk"
]]

df.dropna(subset=["risk"], inplace=True)

# ========================= HANDLE MISSING VALUES =========================
df = df.fillna(df.median(numeric_only=True))

# ========================= SPLIT FEATURES & TARGET =========================
X = df.drop("risk", axis=1)
y = df["risk"]

# ========================= TRAIN-TEST SPLIT =========================
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========================= SMOTE-TOMEK =========================
from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=42)
Xtr_resampled, ytr_resampled = smt.fit_resample(Xtr, ytr)

# ========================= SCALE FEATURES =========================
scaler = StandardScaler()
Xtr_scaled = scaler.fit_transform(Xtr_resampled)
Xte_scaled = scaler.transform(Xte)

# ========================= MODEL =========================
model = RandomForestClassifier(
    n_estimators=45,
    max_depth=2,
    min_samples_split=15,
    min_samples_leaf=30,
    max_features="sqrt",
    class_weight={0:1, 1:1.3},
    random_state=42,
    bootstrap=True
)

# ========================= CROSS-VALIDATION =========================
from sklearn.model_selection import cross_validate

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': make_scorer(recall_score, zero_division=0),
    'f1': make_scorer(f1_score, zero_division=0)
}

# IMPORTANT: use ORIGINAL training data (no SMOTE here)
cv_results = cross_validate(model, Xtr, ytr, cv=5, scoring=scoring)

print('Cross-Validation Metrics (5 folds):')
for metric in scoring.keys():
    scores = cv_results[f'test_{metric}']
    print(f'{metric.capitalize()}: Mean={scores.mean():.3f}, Std={scores.std():.3f}')

# ========================= TRAIN FINAL MODEL =========================
model.fit(Xtr_scaled, ytr_resampled)

# ========================= PREDICTIONS =========================
ytrain_pred = model.predict(Xtr_scaled)
ytest_pred = model.predict(Xte_scaled)

# ========================= METRICS =========================
train_metrics = {
    'accuracy': accuracy_score(ytr_resampled, ytrain_pred),
    'precision': precision_score(ytr_resampled, ytrain_pred, zero_division=0),
    'recall': recall_score(ytr_resampled, ytrain_pred, zero_division=0),
    'f1': f1_score(ytr_resampled, ytrain_pred, zero_division=0)
}

test_metrics = {
    'accuracy': accuracy_score(yte, ytest_pred),
    'precision': precision_score(yte, ytest_pred, zero_division=0),
    'recall': recall_score(yte, ytest_pred, zero_division=0),
    'f1': f1_score(yte, ytest_pred, zero_division=0)
}

# ========================= SAVE =========================
joblib.dump(model, 'maternal_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(train_metrics, 'train_metrics.pkl')
joblib.dump(test_metrics, 'metrics.pkl')
joblib.dump(yte, 'ytest.pkl')
joblib.dump(ytest_pred, 'ypred.pkl')

print('\nFinal Model Trained and Saved ✓')
print(f'Train metrics: {train_metrics}')
print(f'Test metrics: {test_metrics}')
