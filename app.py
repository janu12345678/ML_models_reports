import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import os

# Set page config
st.set_page_config(page_title="Dry Bean Classification", layout="wide")

st.title("Dry Bean Classification App")
st.markdown("""
This application demonstrates multiple Machine Learning models for classifying dry beans into 7 different types based on their features.
Dataset: UCI Machine Learning Repository [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset).
""")

# Sidebar for Model Selection
st.sidebar.header("Model Selection")
model_options = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
]
selected_model_name = st.sidebar.selectbox("Choose a Classification Model", model_options)

# Load Model
@st.cache_resource
def load_model(model_name):
    filename = f"model/{model_name.replace(' ', '_').lower()}.pkl"
    try:
        model = joblib.load(filename)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found: {filename}.\nPlease train the models first using 'python model/train_models.py'.")
        return None

# Load Scaler
@st.cache_resource
def load_scaler():
    try:
        return joblib.load('model/scaler.pkl')
    except FileNotFoundError:
        st.error("Scaler not found. Please ensure 'model/scaler.pkl' exists.")
        return None

# Load Label Encoder
@st.cache_resource
def load_label_encoder():
    try:
        return joblib.load('model/label_encoder.pkl')
    except FileNotFoundError:
        return None

model = load_model(selected_model_name)
scaler = load_scaler()
le = load_label_encoder()

# Main Content
st.header("Upload Test Data")
uploaded_file = st.file_uploader("Upload your input CSV file (features + 'Class' column)", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Sample:")
        st.dataframe(data)
        
        # Check if target column exists for evaluation
        target_col = 'Class' 
        
        if target_col in data.columns:
            X_test = data.drop(columns=[target_col])
            y_test = data[target_col]
            
            # Scale data
            if scaler:
                try:
                    X_test_scaled = scaler.transform(X_test)
                except ValueError as e:
                    st.error(f"Feature mismatch: {e}")
                    st.stop()
            else:
                st.stop()
            
            if model:
                # Predictions
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None
                
                # Metrics (Multiclass)
                st.subheader("Evaluation Metrics (Weighted Average)")
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                
                # AUC for multiclass
                auc_score = "N/A"
                if y_prob is not None:
                    try:
                        auc_score = f"{roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted'):.4f}"
                    except:
                        pass
                col2.metric("AUC Score", auc_score)
                
                col3.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")
                
                col4, col5, col6 = st.columns(3)
                col4.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.4f}")
                col5.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.4f}")
                col6.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8,6))
                
                # Decode labels for plot if encoder available
                if le:
                     labels = le.classes_
                     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=labels, yticklabels=labels)
                else:
                     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Classification Report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
                
        else:
            st.warning(f"Uploaded CSV does not contain '{target_col}' column. Evaluation metrics cannot be calculated.")
            
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
