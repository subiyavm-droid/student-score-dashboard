import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import os

st.set_page_config(page_title="Student Performance Intelligence System", layout="wide")

st.title("🎓 STUDENT EXAM SCORE PREDICTION AND AI SUGGESTION SYSTEM")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("StudentsPerformance.csv")

# ---------------- DATA ENCODING ----------------
df_encoded = df.copy()
label_encoders = {}

for col in df_encoded.columns:
    if df_encoded[col].dtype == "object":
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

# ---------------- MODEL TRAINING ----------------
X = df_encoded.drop(["math score", "reading score", "writing score"], axis=1)
y = df_encoded[["math score", "reading score", "writing score"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.subheader("📊 Model Performance")
st.write("Overall R² Score:", round(r2_score(y_test, y_pred), 3))

# ---------------- GENERATE PREDICTIONS FOR ALL STUDENTS ----------------
full_predictions = model.predict(X)

full_output = df.copy()

full_output["Predicted Math Score"] = full_predictions[:, 0]
full_output["Predicted Reading Score"] = full_predictions[:, 1]
full_output["Predicted Writing Score"] = full_predictions[:, 2]

full_output["Total Score"] = (
    full_output["Predicted Math Score"] +
    full_output["Predicted Reading Score"] +
    full_output["Predicted Writing Score"]
)

full_output["Performance Index"] = full_output["Total Score"] / 3

def categorize(score):
    if score >= 80:
        return "Excellent"
    elif score >= 50:
        return "Average"
    else:
        return "Poor"

full_output["Performance Category"] = full_output["Performance Index"].apply(categorize)

full_output.to_csv("prediction_output_full.csv", index=False)

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("Enter Student Details")

gender = st.sidebar.selectbox("Gender", df["gender"].unique())
race = st.sidebar.selectbox("Race/Ethnicity", df["race/ethnicity"].unique())
parent_edu = st.sidebar.selectbox("Parental Education", df["parental level of education"].unique())
lunch = st.sidebar.selectbox("Lunch Type", df["lunch"].unique())
prep = st.sidebar.selectbox("Test Preparation Course", df["test preparation course"].unique())

if st.sidebar.button("Predict Performance"):

    input_df = pd.DataFrame({
        "gender": [gender],
        "race/ethnicity": [race],
        "parental level of education": [parent_edu],
        "lunch": [lunch],
        "test preparation course": [prep]
    })

    for col in input_df.columns:
        input_df[col] = label_encoders[col].transform(input_df[col])

    prediction = model.predict(input_df)

    math_pred = round(prediction[0][0], 2)
    reading_pred = round(prediction[0][1], 2)
    writing_pred = round(prediction[0][2], 2)

    total_score = round(math_pred + reading_pred + writing_pred, 2)
    avg_score = round(total_score / 3, 2)

    if avg_score >= 80:
        performance = "Excellent"
    elif avg_score >= 50:
        performance = "Average"
    else:
        performance = "Poor"

    # ---------------- DISPLAY RESULTS ----------------
    st.subheader("📈 Prediction Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Math Score", math_pred)
    col2.metric("Reading Score", reading_pred)
    col3.metric("Writing Score", writing_pred)

    st.metric("Total Score", total_score)
    st.metric("Performance Level", performance)

    # ---------------- AI SUGGESTIONS ----------------
    st.subheader("🧠 AI-Based Improvement Suggestions")

    suggestions = []

    if math_pred < 50:
        suggestions.append("Practice daily math problem-solving and focus on weak concepts.")
    if reading_pred < 50:
        suggestions.append("Improve reading comprehension through newspapers and storybooks.")
    if writing_pred < 50:
        suggestions.append("Work on grammar, vocabulary and structured writing practice.")
    if avg_score >= 80:
        suggestions.append("Excellent performance! Prepare for competitive exams.")

    if not suggestions:
        suggestions.append("Maintain consistency and revise weekly for improvement.")

    for s in suggestions:
        st.write("•", s)

    # ---------------- PREDICTED SCORE VISUALIZATION ----------------
    st.subheader("📊 Predicted Score Visualization")

    subjects = ["Math", "Reading", "Writing"]
    predicted_scores = [math_pred, reading_pred, writing_pred]

    fig_pred, ax_pred = plt.subplots()
    ax_pred.bar(subjects, predicted_scores)
    ax_pred.set_ylim(0, 100)
    ax_pred.set_ylabel("Predicted Score")
    ax_pred.set_title("Predicted Subject Scores")
    st.pyplot(fig_pred)

    # ---------------- COMPARISON WITH DATASET AVERAGE ----------------
    st.subheader("📈 Predicted vs Dataset Average")

    dataset_avg = [
        df["math score"].mean(),
        df["reading score"].mean(),
        df["writing score"].mean()
    ]

    x = np.arange(len(subjects))
    width = 0.35

    fig_compare, ax_compare = plt.subplots()
    ax_compare.bar(x - width/2, predicted_scores, width, label="Predicted")
    ax_compare.bar(x + width/2, dataset_avg, width, label="Dataset Average")

    ax_compare.set_xticks(x)
    ax_compare.set_xticklabels(subjects)
    ax_compare.set_ylim(0, 100)
    ax_compare.set_ylabel("Score")
    ax_compare.set_title("Predicted vs Average Comparison")
    ax_compare.legend()

    st.pyplot(fig_compare)

    # ---------------- PDF REPORT ----------------
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Student Performance Prediction Report", styles["Heading1"]))
    elements.append(Spacer(1, 12))

    data = [
        ["Subject", "Predicted Score"],
        ["Math", math_pred],
        ["Reading", reading_pred],
        ["Writing", writing_pred],
        ["Total", total_score],
        ["Performance", performance]
    ]

    table = Table(data)
    elements.append(table)

    doc.build(elements)

    st.download_button(
        label="📄 Download Prediction Report (PDF)",
        data=buffer.getvalue(),
        file_name="Student_Performance_Report.pdf",
        mime="application/pdf"
    )

    # ---------------- SAVE FOR POWER BI ----------------
    output_data = pd.DataFrame({
        "gender": [gender],
        "race/ethnicity": [race],
        "parental level of education": [parent_edu],
        "lunch": [lunch],
        "test preparation course": [prep],
        "Predicted Math Score": [math_pred],
        "Predicted Reading Score": [reading_pred],
        "Predicted Writing Score": [writing_pred],
        "Total Score": [total_score],
        "Performance Index": [avg_score],
        "Performance Category": [performance]
    })

    file_path = "prediction_output.csv"

    if os.path.exists(file_path):
        existing = pd.read_csv(file_path)
        updated = pd.concat([existing, output_data], ignore_index=True)
        updated.to_csv(file_path, index=False)
    else:
        output_data.to_csv(file_path, index=False)

    st.success("Prediction saved successfully for Power BI Dashboard!")

# ---------------- FEATURE IMPORTANCE ----------------
st.subheader("🔥 Feature Importance (Math Model)")

rf_model = model.estimators_[0]
importances = rf_model.feature_importances_
features = X.columns

fig, ax = plt.subplots()
ax.barh(features, importances)
ax.set_xlabel("Importance")
st.pyplot(fig)
