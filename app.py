import streamlit as st
import joblib
import numpy as np

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Multi Disease Prediction",
    page_icon="🩺",
    layout="centered"
)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    return joblib.load("all_disease_models.pkl")

models = load_models()

# ---------------- SIDEBAR ----------------
st.sidebar.title("Disease Prediction System")
page = st.sidebar.radio(
    "Select Disease",
    ["Diabetes", "Heart Disease", "Kidney Disease"]
)

st.title("🩺 Multi-Disease Prediction App")

# ---------------- DIABETES ----------------
if page == "Diabetes":
    st.header("Diabetes Prediction")

    pregnancies = st.number_input("Pregnancies", 0, 20, 0)
    glucose = st.number_input("Glucose", 0, 300, 120)
    bp = st.number_input("Blood Pressure", 0, 200, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 100.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 5.0, 0.5)
    age = st.number_input("Age", 0, 120, 30)

    if st.button("Predict Diabetes"):
        data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        scaled = models["diabetes_scaler"].transform(data)
        prediction = models["diabetes_model"].predict(scaled)[0]

        st.success("🟢 Positive" if prediction == 1 else "🟢 Negative")

# ---------------- HEART DISEASE ----------------
elif page == "Heart Disease":
    st.header("Heart Disease Prediction")

    age = st.number_input("Age", 0, 120, 50)
    sex = st.radio("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting BP", 0, 250, 120)
    chol = st.number_input("Cholesterol", 0, 600, 200)
    fbs = st.radio("Fasting Blood Sugar > 120", ["Yes", "No"])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", 0, 250, 150)
    exang = st.radio("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Major Vessels", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [1, 2, 3])

    if st.button("Predict Heart Disease"):
        sex = 1 if sex == "Male" else 0
        fbs = 1 if fbs == "Yes" else 0
        exang = 1 if exang == "Yes" else 0

        data = np.array([[age, sex, cp, trestbps, chol, fbs,
                          restecg, thalach, exang, oldpeak,
                          slope, ca, thal]])

        scaled = models["heart_scaler"].transform(data)
        prediction = models["heart_model"].predict(scaled)[0]

        st.success("🔴 Positive" if prediction == 1 else "🟢 Negative")

# ---------------- KIDNEY DISEASE ----------------
elif page == "Kidney Disease":
    st.header("Kidney Disease Prediction")

    feature_count = models["kidney_scaler"].mean_.shape[0]
    inputs = []

    for i in range(feature_count):
        inputs.append(st.number_input(f"Feature {i+1}", value=0.0))

    if st.button("Predict Kidney Disease"):
        data = np.array(inputs).reshape(1, -1)
        scaled = models["kidney_scaler"].transform(data)
        prediction = models["kidney_model"].predict(scaled)[0]

        st.success("🔴 CKD Detected" if prediction == 1 else "🟢 Not CKD")
