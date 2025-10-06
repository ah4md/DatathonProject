# 📊 Diabetes Analytics Project Report

---

## 👥 Team Details

| Name | Roll No | Role | Contribution |
|------|----------|------|--------------|
| **Ahamed Mulla** | 46 | Preprocessing & Dashboard Integration | Data cleaning, preprocessing scripts, Streamlit dashboard |
| **Yogesh Mane** | 47 | Visualization & EDA | EDA, visualizations, data insights |
| **Ziyad Kakhandkikar** | 48 | Modeling & Model Evaluation | Model training, evaluation metrics, feature importance |

---

## 🩺 1. Introduction

Diabetes is a chronic metabolic disease affecting millions worldwide, often resulting from poor insulin regulation.  
Early detection can significantly improve patient outcomes through lifestyle modification and early treatment.  

This project applies **Data Analytics** techniques to a real-world *unclean clinical dataset* from Kaggle, aiming to:

- Clean and preprocess noisy healthcare data  
- Explore patterns and relationships among clinical variables  
- Build predictive models to identify diabetic patients  
- Present findings through an **interactive Streamlit dashboard**

This Datathon project represents a full data analytics pipeline — from raw data handling to actionable insights and visualization.

---

## 🎯 2. Problem Statement

The dataset contains inconsistencies such as missing values, invalid entries (zeros in lab results), and mixed categorical labels.  
The main problem is to **process this unclean dataset** to build a reliable diabetes prediction model and uncover clinical insights.

**Objectives:**
1. Perform comprehensive data cleaning and transformation  
2. Conduct Exploratory Data Analysis (EDA) to understand trends and correlations  
3. Train and compare multiple classification models  
4. Create an interactive visualization dashboard for result interpretation  
5. Derive actionable insights relevant to medical decision-making  

---

## 📚 3. Dataset Description

- **Source:** [Kaggle – Diabetes Unclean Dataset](https://www.kaggle.com/datasets/ryugorocks/diabetes-unclean)  
- **Type:** Tabular dataset containing clinical health measurements  
- **Size:** ~1009 rows (patients), ~14 columns  
- **Target Variable:** `CLASS` (1 = Diabetic, 0 = Non-Diabetic)  

### 🔸 Features

| Feature | Description |
|----------|-------------|
| AGE | Patient’s age |
| HbA1c | Hemoglobin A1c percentage |
| BMI | Body Mass Index |
| Chol | Total Cholesterol |
| HDL, LDL, VLDL, TG | Lipid profile indicators |
| CLASS | Diagnosis result (target) |

### ⚠️ Data Issues Identified

- Missing values in key columns  
- Zero values representing missing entries  
- Inconsistent target labels (“Y”, “Yes”, “Diabetic”)  
- Irrelevant columns (`ID`, `No_Pation`, `Gender`, `Urea`, `Cr`)  

**Sample of Raw Dataset:**
<img width="940" height="348" alt="image" src="https://github.com/user-attachments/assets/802acd08-309f-456e-b47e-db9bfc9d1a8c" />

---

## ⚙️ 4. Methodology

### 4.1 Data Preprocessing

Steps performed:

1. Removed unnecessary columns: `ID`, `No_Pation`, `Gender`, `Urea`, `Cr`.  
2. Replaced invalid zeros in (`HbA1c`, `Chol`, `TG`, `HDL`, `LDL`, `VLDL`, `BMI`) with `NaN` and imputed median values.  
3. Filled missing `AGE` values with median.  
4. Standardized and encoded `CLASS` labels: `['Y', 'YES', 'DIABETIC'] → 1`, others → 0.  
5. Scaled numerical features using **StandardScaler** for model training.

**Visualization – Before vs After Cleaning:**
<img width="2400" height="1500" alt="image" src="https://github.com/user-attachments/assets/32c38be9-d40d-412a-a599-62e81ba530ce" />

---

### 4.2 Exploratory Data Analysis (EDA)

EDA was performed using **Pandas**, **Matplotlib**, **Seaborn**, and **Plotly**.

- **Class Distribution:** Checked dataset balance between diabetic and non-diabetic cases.  
- **Scatterplots:** `BMI vs HbA1c` (color-coded by diabetes class).  
- **Trend Analysis:** Average cholesterol levels across different age groups.  
- **Correlation Heatmap:** Identified strong relationships between HbA1c, BMI, and diabetes.

**Visualizations:**

<img width="1800" height="1200" alt="image" src="https://github.com/user-attachments/assets/353dd23b-007e-47bb-8bf6-c0df7a18000b" />  
<img width="2100" height="1500" alt="image" src="https://github.com/user-attachments/assets/fe2f7028-c7ca-41cd-8886-75566a99bb87" />  
<img width="2400" height="1800" alt="image" src="https://github.com/user-attachments/assets/a8042ee9-248b-439f-8ea2-e2a26d3657f0" />

---

### 4.3 Modeling

Two supervised learning models were implemented:

| Model | Library | Description |
|--------|----------|-------------|
| Logistic Regression | Scikit-learn | Linear model for binary classification |
| Random Forest Classifier | Scikit-learn | Ensemble model combining multiple decision trees |

**Data Split:** 80% training, 20% testing (stratified).  
**Evaluation Metrics:** Accuracy, ROC-AUC, Confusion Matrix, Classification Report.

**Model Visualizations:**

<img width="1500" height="1200" alt="image" src="https://github.com/user-attachments/assets/a400096b-8dca-4c86-affc-acec9875ea80" />  
<img width="2400" height="1500" alt="image" src="https://github.com/user-attachments/assets/aa7d4a8f-1f46-4bdb-b497-fc381cedc05f" />

---

### 4.4 Dashboard Development

The final dashboard was built using **Streamlit**, integrating all analytical outputs into a clean, interactive interface.

#### 🔹 Dashboard Features

- **Sidebar Filters** (for age range selection)  
  <img width="235" height="291" alt="image" src="https://github.com/user-attachments/assets/df90e39a-155c-4c2a-b247-7915402ea14f" />

- **Interactive EDA Plots** (class distribution, BMI vs HbA1c, cholesterol trend)  
  <img width="1625" height="998" alt="image" src="https://github.com/user-attachments/assets/8e5f86cb-a853-4e71-9d55-fa1ce689b477" />  
  <img width="1555" height="462" alt="image" src="https://github.com/user-attachments/assets/04c78b3d-bfda-4156-8e63-5973683028fc" />

- **Custom Feature Scatterplot**  
  <img width="1579" height="499" alt="image" src="https://github.com/user-attachments/assets/85638d12-1ec9-471c-9075-a7e84a267a2c" />

- **Model Performance Table & Metrics**  
  <img width="1566" height="244" alt="image" src="https://github.com/user-attachments/assets/79c97150-9df3-4251-8613-d5bdea384a95" />

- **Feature Importance & Insights Section**  
  <img width="1564" height="741" alt="image" src="https://github.com/user-attachments/assets/f4504901-e442-43ee-81e2-410b87481746" />

---

## 📈 5. Results & Analysis

### 5.1 Model Performance

| Model | Accuracy | ROC-AUC |
|--------|-----------|---------|
| Logistic Regression | 0.9604 | 0.9872 |
| Random Forest | **0.9851** | **1.0000** |

- Random Forest achieved higher accuracy and perfect ROC-AUC.  
- Logistic Regression performed well but lacked non-linear feature learning.

### 5.2 Key Observations

- **High HbA1c** and **BMI** strongly indicate diabetes risk.  
- **Cholesterol** levels increase with age among diabetic patients.  
- Dataset shows moderate class imbalance affecting recall.

---

## 💡 6. Discussion & Insights

- The model validates medical understanding — **HbA1c** and **BMI** are strong diabetes predictors.  
- The dashboard provides visual tools for clinicians to interpret data intuitively.  
- Random Forest proved more robust due to its non-linear decision boundaries.  
- Future improvements can address dataset imbalance and scalability.

---

## ⚠️ 7. Limitations & Future Enhancements

- Dataset size is limited — generalization might be constrained.  
- Class imbalance could bias predictions; resampling techniques like **SMOTE** may help.  
- Only two models tested — future exploration could include **XGBoost**, **SVM**, or **Neural Networks**.  
- Dashboard can be hosted publicly (Streamlit Cloud / Render / Hugging Face Spaces).

---

## 🧩 8. Conclusion

This Datathon project successfully implemented the complete analytics pipeline:

- Cleaned and standardized real-world health data  
- Conducted meaningful exploratory data analysis  
- Built accurate predictive models  
- Presented results interactively via a Streamlit dashboard  

**Impact:**  
The project demonstrates how data-driven analytics can support early diabetes detection and empower healthcare decision-making.

---

## 📚 9. References

1. Kaggle – Diabetes Unclean Dataset  
2. Scikit-learn Documentation  
3. Streamlit Documentation  
4. Seaborn & Matplotlib Libraries  

---

## 🖥️ 10. Instructions to Run

```bash
# Clone the repository
git clone https://github.com/ah4md/DatathonProject.git
cd DatathonProject

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard.py
