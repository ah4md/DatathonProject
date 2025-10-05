# dashboard.py
# Run with: streamlit run app/dashboard.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
import os

# Create outputs folder if not exists
os.makedirs("outputs", exist_ok=True)

# =========================
# Load dataset
# =========================
df = pd.read_csv("../data/cleaned/cleaned_diabetes.csv")


# =========================
# Streamlit Dashboard
# =========================
st.set_page_config(page_title="Diabetes Analytics Dashboard", layout="wide")
st.title("📊 Diabetes Analytics Dashboard")
st.write("This dashboard provides insights into the diabetes dataset using Python and Streamlit.")

# Sidebar filters
st.sidebar.header("Filters")
age_range = st.sidebar.slider("Select Age Range:", int(df['AGE'].min()), int(df['AGE'].max()), (20, 60))
filtered_df = df[(df['AGE'] >= age_range[0]) & (df['AGE'] <= age_range[1])]

# =========================
# EDA Visualizations
# =========================
st.subheader("Class Distribution")
fig1 = px.histogram(filtered_df, x="CLASS", color="CLASS", barmode="group",
                    labels={"CLASS": "Diabetes (1=Yes, 0=No)"})
st.plotly_chart(fig1, use_container_width=True)
fig1.write_html("outputs/class_distribution.html")

st.subheader("BMI vs HbA1c")
fig2 = px.scatter(filtered_df, x="BMI", y="HbA1c", color=filtered_df["CLASS"].astype(str),
                  labels={"color": "Class"}, opacity=0.7)
st.plotly_chart(fig2, use_container_width=True)
fig2.write_html("outputs/bmi_vs_hba1c.html")

st.subheader("Average Cholesterol by Age")
chol_age = filtered_df.groupby("AGE")["Chol"].mean().reset_index()
fig3 = px.line(chol_age, x="AGE", y="Chol", markers=True)
st.plotly_chart(fig3, use_container_width=True)
fig3.write_html("outputs/cholesterol_by_age.html")

# Custom scatterplot
st.subheader("Custom Scatterplot")
x_col = st.sidebar.selectbox("X-axis Feature", df.columns[:-1])
y_col = st.sidebar.selectbox("Y-axis Feature", df.columns[:-1])
fig_custom = px.scatter(filtered_df, x=x_col, y=y_col, color=filtered_df["CLASS"].astype(str),
                        labels={"color": "Class"}, opacity=0.7)
st.plotly_chart(fig_custom, use_container_width=True)
fig_custom.write_html(f"outputs/custom_scatter_{x_col}_{y_col}.html")

# =========================
# Machine Learning Models
# =========================
st.header("🤖 Machine Learning Models")
X = df.drop("CLASS", axis=1)
y = df["CLASS"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2,
                                                    random_state=42, stratify=y)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_pred_rf_prob = rf.predict_proba(X_test)[:, 1]

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_pred_lr_prob = lr.predict_proba(X_test)[:, 1]

# =========================
# Model Evaluation
# =========================
st.subheader("Model Performance Metrics")
acc_rf = rf.score(X_test, y_test)
acc_lr = lr.score(X_test, y_test)
roc_rf = roc_auc_score(y_test, y_pred_rf_prob)
roc_lr = roc_auc_score(y_test, y_pred_lr_prob)

results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [acc_lr, acc_rf],
    "ROC-AUC": [roc_lr, roc_rf]
})
st.dataframe(results)
results.to_csv("outputs/model_performance.csv", index=False)

# Confusion Matrix for Random Forest
st.subheader("Confusion Matrix (Random Forest)")
fig5, ax = plt.subplots()
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test, ax=ax, cmap="Blues")
st.pyplot(fig5)
fig5.savefig("outputs/confusion_matrix.png")

# Classification Report (RF)
st.subheader("Classification Report (Random Forest)")
report = classification_report(y_test, y_pred_rf)
st.text(report)
with open("outputs/classification_report.txt", "w") as f:
    f.write(report)

# Feature Importance
st.subheader("Feature Importance (Random Forest)")
importances = pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_})
importances = importances.sort_values(by="Importance", ascending=False)

fig4 = px.bar(importances, x="Feature", y="Importance", color="Importance", color_continuous_scale="Viridis")
st.plotly_chart(fig4, use_container_width=True)
fig4.write_html("outputs/feature_importance.html")

# =========================
# Insights
# =========================
st.markdown("---")
st.write("✅ Insights:")
st.write("- HbA1c and BMI are strong predictors of diabetes.")
st.write("- Random Forest performed slightly better than Logistic Regression.")
st.write("- This model can help in early diagnosis and lifestyle intervention recommendations.")
