import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import joblib
import os

# 1. Load Dataset (Dry Bean Dataset - ID 602)
print("Fetching Dry Bean Dataset...")
dry_bean_dataset = fetch_ucirepo(id=602)
X = dry_bean_dataset.data.features
y = dry_bean_dataset.data.targets

print(f"Dataset Shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Encode Target (Multiclass)
le = LabelEncoder()
y_encoded = le.fit_transform(y.values.ravel())
y = pd.Series(y_encoded, name='Class') # 'Class' is the target name in UCI

# Save LabelEncoder for app
joblib.dump(le, 'model/label_encoder.pkl')

# 2. Split Data (85/15)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# Save test data for Streamlit upload feature
test_data = X_test.copy()
test_data['Class'] = y_test # Using encoded labels for logic consistency, or could use original if we decode. Let's keep numeric for ML flow
test_data.to_csv('data/dry_bean_test_data.csv', index=False)
print("Test data saved to data/dry_bean_test_data.csv")

# 3. Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'model/scaler.pkl')

# 4. Define Models
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=2000, multi_class='multinomial'),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42) # mlogloss for multiclass
}

# 5. Train and Evaluate
results = []

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None
    
    # Calculate Metrics (Multiclass: weighted average)
    acc = accuracy_score(y_test, y_pred)
    # AUC for multiclass requires ovr/ovo
    auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted') if y_prob is not None else "N/A"
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "AUC": auc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc
    })
    
    # Save Model
    joblib.dump(model, f'model/{name.replace(" ", "_").lower()}.pkl')

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print("\nModel Evaluation Results:")
print(results_df)

# Save results for reference explanation in README
results_df.to_csv('data/model_results.csv', index=False)
print("Training completed. Models and results saved.")
