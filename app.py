import streamlit as st
import joblib
import numpy as np

# Load all models
models = joblib.load("all_disease_models.pkl")

# Sidebar navigation
st.sidebar.title("Disease Prediction System")
page = st.sidebar.radio("Select Disease", ["Diabetes", "Heart Disease", "Kidney Disease"])

st.title("🩺 Multi-Disease Prediction App")

# ----------------- DIABETES PAGE -----------------
if page == "Diabetes":
    st.header("Diabetes Prediction")
    st.write("Enter patient details to predict Diabetes:")

    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
    bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, value=0.5)
    age = st.number_input("Age", min_value=0, max_value=120, value=30)

    if st.button("Predict Diabetes"):
        input_data = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
        prediction = models["diabetes_model"].predict(models["diabetes_scaler"].transform([input_data]))[0]
        result = "Positive" if prediction == 1 else "Negative"
        st.success(f"Diabetes Prediction: {result}")

# ----------------- HEART DISEASE PAGE -----------------
elif page == "Heart Disease":
    st.header("Heart Disease Prediction")
    st.write("Enter patient details to predict Heart Disease:")

    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    sex = st.radio("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (0–3)", [0,1,2,3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=250, value=120)
    chol = st.number_input("Serum Cholesterol", min_value=0, max_value=600, value=200)
    fbs = st.radio("Fasting Blood Sugar >120 mg/dl", ["Yes", "No"])
    restecg = st.selectbox("Resting ECG (0–2)", [0,1,2])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=0, max_value=250, value=150)
    exang = st.radio("Exercise Induced Angina", ["Yes","No"])
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope of ST Segment", [0,1,2])
    ca = st.selectbox("Number of Major Vessels (0–3)", [0,1,2,3])
    thal = st.selectbox("Thalassemia (1–3)", [1,2,3])

    if st.button("Predict Heart Disease"):
        sex_val = 1 if sex=="Male" else 0
        fbs_val = 1 if fbs=="Yes" else 0
        exang_val = 1 if exang=="Yes" else 0

        input_data = [age, sex_val, cp, trestbps, chol, fbs_val, restecg, thalach, exang_val, oldpeak, slope, ca, thal]
        prediction = models["heart_model"].predict(models["heart_scaler"].transform([input_data]))[0]
        result = "Positive" if prediction == 1 else "Negative"
        st.success(f"Heart Disease Prediction: {result}")

# ----------------- KIDNEY DISEASE PAGE -----------------
elif page == "Kidney Disease":
    st.header("Kidney Disease Prediction")
    st.write("Enter patient numeric details to predict Kidney Disease:")

    # Get numeric feature names from the model/scaler
    feature_names = models["kidney_scaler"].mean_.shape[0]
    kidney_inputs = []
    for i in range(feature_names):
        val = st.number_input(f"Feature {i+1}", value=0.0)
        kidney_inputs.append(val)

    if st.button("Predict Kidney Disease"):
        input_data = np.array(kidney_inputs).reshape(1,-1)
        prediction = models["kidney_model"].predict(models["kidney_scaler"].transform(input_data))[0]
        result = "CKD" if prediction == 1 else "Not CKD"
        st.success(f"Kidney Disease Prediction: {result}")

