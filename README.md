# Credit Risk Modeling

This project builds a machine learning model to predict credit risk using LendingClub loan data (2007–2014).  
The model applies a **Random Forest Classifier** with preprocessing pipelines, data balancing using **SMOTE**, and evaluation through **ROC-AUC** and a confusion matrix.

---

## Features
- **Data Preprocessing**
  - Handle missing values
  - Convert `emp_length` and `term` to numeric
  - Label encoding for categorical columns
  - Create new feature `credit_history_age`
- **Data Balancing**
  - SMOTE for handling imbalanced classes
- **Model**
  - Random Forest with basic hyperparameters
  - Integrated pipeline using scikit-learn + imbalanced-learn
- **Evaluation**
  - Classification Report
  - ROC-AUC Score (multi-class OVR)
  - Confusion Matrix (saved as `confusion_matrix_75pct.png`)
- **Model Saving**
  - Export trained model as `credit_risk_model_75pct.pkl` using joblib

---

## Requirements
Install dependencies with:
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib
