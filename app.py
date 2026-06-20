import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load Model
with open("model/student_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load Dataset
df = pd.read_csv("data/students.csv")

# Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Title
st.title("🎓 Student Performance Predictor Dashboard")
st.markdown("---")

# Sidebar Inputs
st.sidebar.header("Student Details")

study_hours = st.sidebar.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=12.0,
    value=5.0
)

attendance = st.sidebar.number_input(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=80
)

previous_marks = st.sidebar.number_input(
    "Previous Marks",
    min_value=0,
    max_value=100,
    value=70
)

predict_button = st.sidebar.button("Predict")

# Dashboard Columns
col1, col2 = st.columns(2)

# Prediction Section
if predict_button:

    prediction = model.predict(
        pd.DataFrame(
            [[study_hours, attendance, previous_marks]],
            columns=["StudyHours", "Attendance", "PreviousMarks"]
        )
    )[0]

    with col1:
        st.metric(
            label="Predicted Final Marks",
            value=f"{prediction:.2f}"
        )

    with col2:

        if prediction >= 85:
            st.success("🏆 Performance: Excellent")

        elif prediction >= 70:
            st.info("👍 Performance: Good")

        elif prediction >= 50:
            st.warning("⚠ Performance: Average")

        else:
            st.error("❌ Performance: Needs Improvement")

    st.markdown("---")

    st.subheader("📌 Recommendations")

    if prediction >= 85:
        st.write("✅ Maintain your current study habits")
        st.write("✅ Continue attending classes regularly")
        st.write("✅ Aim for consistency")

    elif prediction >= 70:
        st.write("✅ Study 1 extra hour daily")
        st.write("✅ Revise difficult topics weekly")
        st.write("✅ Maintain attendance above 85%")

    elif prediction >= 50:
        st.write("⚠ Increase attendance")
        st.write("⚠ Focus on weak subjects")
        st.write("⚠ Create a fixed study schedule")

    else:
        st.write("❌ Immediate improvement needed")
        st.write("❌ Create a daily study plan")
        st.write("❌ Seek guidance from teachers")

# Dataset Preview
st.markdown("---")

st.subheader("📊 Dataset Preview")

st.dataframe(df.head())

# Graph 1
st.subheader("📈 Study Hours vs Final Marks")

fig, ax = plt.subplots(figsize=(6, 4))

ax.scatter(
    df["StudyHours"],
    df["FinalMarks"]
)

ax.set_xlabel("Study Hours")
ax.set_ylabel("Final Marks")
ax.set_title("Study Hours vs Final Marks")

st.pyplot(fig)

# Graph 2
st.subheader("📈 Attendance vs Final Marks")

fig2, ax2 = plt.subplots(figsize=(6, 4))

ax2.scatter(
    df["Attendance"],
    df["FinalMarks"]
)

ax2.set_xlabel("Attendance (%)")
ax2.set_ylabel("Final Marks")
ax2.set_title("Attendance vs Final Marks")

st.pyplot(fig2)

# Footer
st.markdown("---")
st.caption("Built using Python, Scikit-Learn, Pandas, Matplotlib and Streamlit")

st.title("🎓 Student Performance Predictor Dashboard")

# KPI Cards
kpi1, kpi2, kpi3 = st.columns(3)

kpi1.metric(
    "Total Students",
    len(df)
)

kpi2.metric(
    "Average Marks",
    round(df["FinalMarks"].mean(), 2)
)

kpi3.metric(
    "Average Attendance",
    round(df["Attendance"].mean(), 2)
)

# Initialize Prediction History
if "history" not in st.session_state:
    st.session_state.history = []

# Correlation Heatmap
st.subheader("🔥 Correlation Heatmap")

corr = df.corr()

fig3, ax3 = plt.subplots(figsize=(6, 5))

heatmap = ax3.imshow(corr)

plt.colorbar(heatmap)

ax3.set_xticks(range(len(corr.columns)))
ax3.set_xticklabels(corr.columns, rotation=45)

ax3.set_yticks(range(len(corr.columns)))
ax3.set_yticklabels(corr.columns)

st.pyplot(fig3)

# Dataset Statistics
st.subheader("📋 Dataset Statistics")
st.write(df.describe())

# Prediction History
st.subheader("🕒 Prediction History")

history_df = pd.DataFrame(
    st.session_state.history
)

st.dataframe(history_df)

# Insights
st.subheader("📚 Insights")

top_student = df["FinalMarks"].max()

st.write(
    f"Highest Final Marks in Dataset: {top_student}"
)

lowest_student = df["FinalMarks"].min()

st.write(
    f"Lowest Final Marks in Dataset: {lowest_student}"
)

