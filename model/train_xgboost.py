import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 1. Load Dataset (Dry Bean - ID 602)
print("Fetching Dry Bean Dataset...")
dry_bean_dataset = fetch_ucirepo(id=602)
X = dry_bean_dataset.data.features
y = dry_bean_dataset.data.targets

# Encode Target
le = LabelEncoder()
y_encoded = le.fit_transform(y.values.ravel())
y = pd.Series(y_encoded, name='Class')

# 2. Split Data (85/15)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# 3. Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, '../model/scaler.pkl')

# 4. Train XGBoost
print("Training XGBoost...")
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Save Model
joblib.dump(model, '../model/xgboost.pkl')
print("Model saved to ../model/xgboost.pkl")
