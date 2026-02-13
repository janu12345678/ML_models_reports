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
| Logistic Regression | [VAL] | [VAL] | [VAL] | [VAL] | [VAL] | [VAL] |
| Decision Tree | [VAL] | [VAL] | [VAL] | [VAL] | [VAL] | [VAL] |
| kNN | [VAL] | [VAL] | [VAL] | [VAL] | [VAL] | [VAL] |
| Naive Bayes | [VAL] | [VAL] | [VAL] | [VAL] | [VAL] | [VAL] |
| Random Forest | [VAL] | [VAL] | [VAL] | [VAL] | [VAL] | [VAL] |
| XGBoost | [VAL] | [VAL] | [VAL] | [VAL] | [VAL] | [VAL] |

## Project Structure
- `app.py`: Streamlit application file.
- `requirements.txt`: Project dependencies.
- `model/`: Directory containing trained models and scaler.
- `model/individual_scripts/`: Independent python scripts for training each model.
- `data/`: Directory containing test data and results.

