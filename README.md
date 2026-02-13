# Breast Cancer Classification App

## Problem Statement
The goal of this project is to classify breast cancer tumors as Malignant or Benign using various Machine Learning algorithms. Early diagnosis of breast cancer can significantly improve survival rates. This project implements multiple classification models to predict the diagnosis based on features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

## Dataset Description
- **Dataset**: Breast Cancer Wisconsin (Diagnostic)
- **Source**: `sklearn.datasets`
- **Instances**: 569
- **Features**: 30 numeric, predictive attributes + the target.
- **Target**: Malignant (0) vs Benign (1).
- **Data Split**: 80% Training, 20% Testing.

## Models Used
The following classification models were implemented and evaluated:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

### Performance Comparison
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest | 0.9561 | 0.9939 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| XGBoost | 0.9474 | 0.9917 | 0.9459 | 0.9722 | 0.9589 | 0.8864 |

## Project Structure
- `app.py`: Streamlit application file.
- `requirements.txt`: Project dependencies.
- `model/`: Directory containing trained models and scaler.
- `data/`: Directory containing test data and results.

## Observations
- **Logistic Regression** achieved the highest accuracy (98.25%) and AUC (0.9954), suggesting that the dataset features have a strong linear relationship with the target.
- **Decision Tree** showed the lowest performance (Accuracy: 91.23%), which is common for single trees due to high variance and potential overfitting.
- **Ensemble Models** (Random Forest and XGBoost) performed robustly with high accuracy (>94%) and AUC scores, validating their effectiveness in handling complex patterns.
- **KNN** also performed very well (95.61%), indicating that local neighborhood information is highly predictive for this dataset.
- Overall, most models achieved high performance, making this a reliable system for breast cancer classification.
