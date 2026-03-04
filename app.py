import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="B2B Client Risk Dashboard", layout="wide")

st.title("B2B Client Risk & Churn Prediction Dashboard")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("B2B_Client_Churn_5000.csv")

df = load_data()

# -----------------------------
# Risk Score Logic
# -----------------------------
df["Risk_Score"] = (
    df["Payment_Delay_Days"] * 0.4 +
    (100 - df["Monthly_Usage"]) * 0.3 +
    (12 - df["Contract_Length"]) * 0.2 +
    df["Support_Tickets"] * 0.1
)

# Risk Category
def risk_category(score):
    if score < 40:
        return "Low Risk"
    elif score < 70:
        return "Medium Risk"
    else:
        return "High Risk"

df["Risk_Category"] = df["Risk_Score"].apply(risk_category)

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

region = st.sidebar.multiselect("Select Region", df["Region"].unique())
industry = st.sidebar.multiselect("Select Industry", df["Industry"].unique())
risk = st.sidebar.multiselect("Select Risk Category", df["Risk_Category"].unique())

filtered_df = df.copy()

if region:
    filtered_df = filtered_df[filtered_df["Region"].isin(region)]
if industry:
    filtered_df = filtered_df[filtered_df["Industry"].isin(industry)]
if risk:
    filtered_df = filtered_df[filtered_df["Risk_Category"].isin(risk)]

# -----------------------------
# KPI Section
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Clients", len(filtered_df))
col2.metric("High Risk Clients", len(filtered_df[filtered_df["Risk_Category"]=="High Risk"]))
col3.metric("Average Revenue", round(filtered_df["Revenue"].mean(),2))

# -----------------------------
# Machine Learning Model
# -----------------------------
df_ml = df.copy()
df_ml["Renewal_Status"] = df_ml["Renewal_Status"].map({"Yes":1,"No":0})

X = df_ml[["Monthly_Usage","Payment_Delay_Days","Contract_Length","Support_Tickets","Revenue"]]
y = df_ml["Renewal_Status"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train,y_train)

pred = model.predict(X_test)
accuracy = accuracy_score(y_test,pred)

col4.metric("Model Accuracy", round(accuracy*100,2))

# -----------------------------
# Visualizations
# -----------------------------
st.subheader("Risk Category Distribution")

fig1, ax1 = plt.subplots()
filtered_df["Risk_Category"].value_counts().plot(kind="bar", ax=ax1)
st.pyplot(fig1)

st.subheader("Industry Wise Risk")

fig2, ax2 = plt.subplots()
pd.crosstab(filtered_df["Industry"], filtered_df["Risk_Category"]).plot(kind="bar", ax=ax2)
st.pyplot(fig2)

st.subheader("Revenue vs Risk Score")

fig3, ax3 = plt.subplots()
ax3.scatter(filtered_df["Revenue"], filtered_df["Risk_Score"])
ax3.set_xlabel("Revenue")
ax3.set_ylabel("Risk Score")
st.pyplot(fig3)

st.subheader("Confusion Matrix")

fig4, ax4 = plt.subplots()
cm = confusion_matrix(y_test,pred)
ax4.imshow(cm)
ax4.set_xlabel("Predicted")
ax4.set_ylabel("Actual")
st.pyplot(fig4)

# -----------------------------
# Top 20 High Risk Clients
# -----------------------------
st.subheader("Top 20 High Risk Clients")

top20 = filtered_df.sort_values(by="Risk_Score", ascending=False).head(20)
st.dataframe(top20)

# -----------------------------
# Retention Strategy Button
# -----------------------------
st.subheader("AI-Based Retention Strategy")

if st.button("Generate Retention Strategy"):
    st.write("• Offer discount for clients with payment delay > 30 days")
    st.write("• Assign dedicated account manager to high complaint clients")
    st.write("• Provide long-term contract incentives")
    st.write("• Offer loyalty rewards to high revenue clients")
    st.write("• Conduct engagement programs for low usage clients")

# -----------------------------
# Responsible AI Section
# -----------------------------
st.subheader("Ethical Implications of Predicting Client Churn")

st.write("""
• Predictive models may contain bias based on industry or region.
• Labeling clients as 'High Risk' may affect relationship decisions.
• Client data privacy must be protected.
• AI predictions should assist decision-making, not replace human judgment.
""")
