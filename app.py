import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="B2B Client Risk Dashboard", layout="wide")
st.title("B2B Client Risk & Churn Prediction Dashboard")

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    return pd.read_csv("B2B_Client_Churn_5000.csv")

df = load_data()

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

st.success("Dataset Loaded Successfully")

# Show column names (for safety)
st.write("Available Columns:", list(df.columns))

# =============================
# AUTO DETECT NUMERIC COLUMNS
# =============================
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 3:
    st.error("Not enough numeric columns to calculate Risk Score")
    st.stop()

# Use first 3 numeric columns dynamically
col1, col2, col3 = numeric_cols[:3]

# =============================
# RISK SCORE (Dynamic)
# =============================
df["Risk_Score"] = (
    df[col1] * 0.4 +
    df[col2] * 0.3 +
    df[col3] * 0.3
)

# Risk category
def risk_category(score):
    if score < df["Risk_Score"].quantile(0.33):
        return "Low Risk"
    elif score < df["Risk_Score"].quantile(0.66):
        return "Medium Risk"
    else:
        return "High Risk"

df["Risk_Category"] = df["Risk_Score"].apply(risk_category)

# =============================
# SIDEBAR FILTERS (Dynamic)
# =============================
st.sidebar.header("Filters")

cat_cols = df.select_dtypes(include="object").columns.tolist()

filtered_df = df.copy()

for col in cat_cols[:2]:  # first two categorical columns
    selected = st.sidebar.multiselect(col, df[col].unique())
    if selected:
        filtered_df = filtered_df[filtered_df[col].isin(selected)]

risk_filter = st.sidebar.multiselect(
    "Risk Category", df["Risk_Category"].unique()
)

if risk_filter:
    filtered_df = filtered_df[filtered_df["Risk_Category"].isin(risk_filter)]

# =============================
# KPI SECTION
# =============================
k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Clients", len(filtered_df))
k2.metric("High Risk Clients",
          len(filtered_df[filtered_df["Risk_Category"]=="High Risk"]))
k3.metric("Average Risk Score",
          round(filtered_df["Risk_Score"].mean(),2))

# =============================
# MACHINE LEARNING (Dynamic)
# =============================
if "Renewal_Status" in df.columns:

    df_ml = df.copy()

    if df_ml["Renewal_Status"].dtype == "object":
        df_ml["Renewal_Status"] = df_ml["Renewal_Status"].map({"Yes":1,"No":0})

    X = df_ml[numeric_cols]
    y = df_ml["Renewal_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(max_depth=4)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    k4.metric("Model Accuracy", str(round(accuracy*100,2)) + "%")

    st.subheader("Confusion Matrix")

    fig_cm, ax_cm = plt.subplots()
    cm = confusion_matrix(y_test, pred)
    ax_cm.imshow(cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

else:
    k4.metric("Model Accuracy", "N/A")

# =============================
# VISUALS
# =============================
st.subheader("Risk Category Distribution")

fig1, ax1 = plt.subplots()
filtered_df["Risk_Category"].value_counts().plot(kind="bar", ax=ax1)
st.pyplot(fig1)

st.subheader("Risk Score vs First Numeric Column")

fig2, ax2 = plt.subplots()
ax2.scatter(filtered_df[col1], filtered_df["Risk_Score"])
ax2.set_xlabel(col1)
ax2.set_ylabel("Risk Score")
st.pyplot(fig2)

# =============================
# TOP 20 HIGH RISK
# =============================
st.subheader("Top 20 High Risk Clients")

top20 = filtered_df.sort_values(
    by="Risk_Score", ascending=False).head(20)

st.dataframe(top20)

# =============================
# RETENTION STRATEGY
# =============================
st.subheader("AI-Based Retention Strategy")

if st.button("Generate Retention Strategy"):
    st.write("• Offer targeted incentives to high-risk clients")
    st.write("• Improve engagement programs")
    st.write("• Assign dedicated support managers")
    st.write("• Provide flexible contract renewal options")

# =============================
# RESPONSIBLE AI
# =============================
st.subheader("Ethical Implications")

st.write("""
• AI models may contain hidden bias.
• Risk labeling should not replace human judgment.
• Client data must be protected.
• Use predictive analytics responsibly.
""")
