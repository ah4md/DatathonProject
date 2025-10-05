# 📊 Diabetes Analytics Project Report

## 👥 Team Members  
- Ahamed Mulla (Roll no: 46)  — (Preprocessing, Modeling, Dashboard Integration)  
- Yogesh Mane (Roll no: 47) — Visualization & EDA  
- Ziyad Kakhandkikar (Roll no: 48)  — Model Evaluation, Report & Presentation  

---

## 1. Introduction & Problem Statement  
Diabetes is a critical chronic disease affecting millions worldwide. Early detection using clinical markers can improve patient outcomes. Our project aims to analyze a real-world diabetes dataset, build predictive models, and derive insights to assist medical decision-making.

Specifically, we handle an “unclean” clinical dataset that includes missing values, inconsistent labels, and invalid entries. The goal is to:  
1. Clean and preprocess data  
2. Perform exploratory data analysis (EDA)  
3. Train classification models to predict diabetes  
4. Visualize results in an interactive dashboard  
5. Derive actionable insights  

---

## 2. Dataset Description  
- **Source:** Kaggle – Diabetes Unclean Dataset  
- **Type:** Tabular clinical data  
- **Size:** ~1009 rows (patients), ~14 columns (features + target)  
- **Features:**  
  AGE, HbA1c, BMI, Cholesterol, HDL, LDL, VLDL, TG, etc.  
- **Target:** `CLASS` (Diabetic = 1, Non-Diabetic = 0)  

**Data Issues Identified:**  
- Missing values and zeros in lab test fields  
- Inconsistent categories in `CLASS` column (Yes, Y, Diabetic, No)  
- Irrelevant/duplicate columns (ID, No_Pation, Gender, Urea, Cr)  

---

## 3. Methodology  

### 3.1 Preprocessing & Cleaning  
- Dropped columns: `ID, No_Pation, Gender, Urea, Cr`  
- For measurement columns (`HbA1c, Chol, TG, HDL, LDL, VLDL, BMI`): replaced 0 values with `NaN`, then imputed with median  
- Imputed missing `AGE` with median  
- Standardized `CLASS` labels to uppercase, stripped whitespace, then encoded: values in `['Y', 'YES', 'DIABETIC']` → 1, else → 0  
- StandardScaled numerical features before modeling  

### 3.2 Exploratory Data Analysis (EDA)  
- Visualized class distribution  
- Scatterplots: BMI vs HbA1c colored by class  
- Trend plots: average cholesterol by age  
- Interactive scatterplot: user may select features for x and y axes  
- Correlation heatmap to examine relationships among variables  

### 3.3 Modeling  
- Split data: **80% training**, **20% testing**, stratified on target  
- Models used:
  - **Random Forest Classifier**  
  - **Logistic Regression**  
- Evaluation metrics:
  - Accuracy  
  - ROC-AUC  
  - Confusion Matrix (for Random Forest)  
  - Classification Report (Precision, Recall, F1)  
- Feature importance (from Random Forest) to identify key predictors  

### 3.4 Dashboard & Visualization  
- Built using **Streamlit**  
- Sections include:
  - Filters (Age Range)  
  - EDA plots (class distribution, BMI vs HbA1c, cholesterol trend)  
  - Custom scatterplot (user feature selection)  
  - Model performance table (Accuracy, ROC-AUC)  
  - Confusion matrix & classification report  
  - Feature importance bar chart  
  - Insights shown as text  

---

## 4. Results & Observations  

### 4.1 Performance Metrics  
| Model                  | Accuracy | ROC-AUC |
|------------------------|----------|---------|
| Logistic Regression    | 0.9603960396039604 | 0.9871722316544047 |
| Random Forest          | 0.9851485148514851 | 1.0 |



- The Random Forest model achieved slightly higher accuracy and ROC-AUC compared to Logistic Regression.  
- Confusion Matrix (Random Forest) shows true positives, false negatives, etc.  
- Classification Report reveals class-wise performance (precision, recall, F1).  

### 4.2 Feature Importance  
The top predictors from Random Forest (sorted by importance):  
1. **HbA1c**  
2. **BMI**  
3. **Cholesterol**  
(Then possibly LDL, HDL, VLDL, TG)  

These match clinical intuition: HbA1c and BMI are strong indicators of diabetes risk.

### 4.3 EDA Insights  
- The diabetic class is less frequent than the non-diabetic class (class imbalance)  
- Higher BMI and higher HbA1c correlate strongly with diabetic class  
- Cholesterol levels tend to rise with age, especially in older age groups  
- Some features show moderate correlation; multicollinearity was minimal  

---

## 5. Discussion & Recommendations  
- The model confirms that **HbA1c and BMI** are key predictors, so in clinical settings, frequent monitoring of these can aid early diagnosis.  
- Logistic Regression performed decently but lacked the nonlinearity capture of Random Forest.  
- The interactive dashboard can help healthcare professionals explore patient data visually and make informed decisions.  
- Because the dataset is limited in size, adding more data (longitudinal patient data) could improve robustness.  

---

## 6. Limitations & Future Work  
- **Dataset size** is small — results may not generalize broadly  
- **Imbalanced classes** may bias the model; techniques like SMOTE could help  
- Only two models tested — future could try **XGBoost, SVM, neural networks**  
- Time-series or longitudinal modeling (if successive measurements available)  
- Deploy dashboard as a web app (Heroku / Streamlit cloud) for real access  

---

## 7. Conclusion  
This project successfully integrated data cleaning, exploratory analysis, model building, and visualization. The Random Forest model provided strong predictive capability, and feature importance confirmed medical expectations. Our dashboard presents an accessible platform for non-technical users to glean insights. The work demonstrates how data analytics can assist in early diagnosis of diabetes.

---

## 8. Instructions to Run  
1. Clone the repository  
2. Install dependencies: `pip install -r requirements.txt`  
3. Run dashboard:  
```bash
streamlit run dashboard/dashboard.py
