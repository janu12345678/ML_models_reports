# Dry Bean Classification App

## Problem Statement
The goal of this project is to classify dry beans into **7 different types** (Seker, Barbunya, Bombay, Cali, Dermason, Horoz, and Sira) based on 16 features describing their form, shape, type, and structure. This is a **Multiclass Classification** problem.

## Dataset Description
- **Dataset**: [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- **Source**: UCI Machine Learning Repository
- **Instances**: 13,611
- **Features**: 16 numeric attributes (Area, Perimeter, MajorAxisLength, etc.)
- **Target**: Class (7 unique bean types)
- **Data Split**: 85% Training, 15% Testing.

## Models Used
The following classification models were implemented and evaluated:
1. Logistic Regression (Multinomial)
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

### Performance Comparison
Results are evaluated using **Weighted Average** for Precision, Recall, and F1-score due to class imbalance.

| ML Model Name | Accuracy | AUC (OvR) | Precision (W) | Recall (W) | F1 (W) | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9216 | 0.9931 | 0.9225 | 0.9216 | 0.9218 | 0.9054 |
| Decision Tree | 0.8923 | 0.9335 | 0.8920 | 0.8923 | 0.8920 | 0.8698 |
| kNN | 0.9167 | 0.9814 | 0.9174 | 0.9167 | 0.9169 | 0.8993 |
| Naive Bayes | 0.9011 | 0.9902 | 0.9041 | 0.9011 | 0.9012 | 0.8812 |
| Random Forest | 0.9212 | 0.9910 | 0.9213 | 0.9212 | 0.9212 | 0.9046 |
| XGBoost | 0.9226 | 0.9937 | 0.9229 | 0.9226 | 0.9228 | 0.9064 |

## Project Structure
- `app.py`: Streamlit application file.
- `requirements.txt`: Project dependencies.
- `model/`: Directory containing trained models and scaler.
- `model/individual_scripts/`: Independent python scripts for training each model.
- `data/`: Directory containing test data and results.

