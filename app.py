import streamlit as st
import joblib

# Load the trained model and scaler
model = joblib.load("cgpa_model.pkl")
scaler = joblib.load("scaler.pkl")

# Website title
st.title("🎓 Student CGPA Prediction")

st.write("Enter the student details below to predict CGPA.")

# Student inputs
study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=0.0,
    max_value=24.0,
    value=7.0
)

internet_usage = st.number_input(
    "Internet Usage Hours",
    min_value=0.0,
    max_value=24.0,
    value=2.0
)

assignments_completed = st.number_input(
    "Assignments Completed",
    min_value=0.0,
    value=5.0
)

previous_score = st.number_input(
    "Previous Score",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

# Prediction button
if st.button("🔮 Predict CGPA"):

    new_student = [[
        study_hours,
        attendance,
        sleep_hours,
        internet_usage,
        assignments_completed,
        previous_score
    ]]

    # Scale the student data
    new_student_scaled = scaler.transform(new_student)

    # Predict CGPA
    prediction = model.predict(new_student_scaled)[0]

    # Keep CGPA between 0 and 10
    prediction = max(0, min(prediction, 10))

    # Show result
    st.success(f"🎉 Predicted CGPA: {prediction:.2f}")