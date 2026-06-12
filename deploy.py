import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart AI Predictor",
    page_icon="❤️",
    layout="wide"
)

# ── Load artifacts ───────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model           = pickle.load(open("heart_disease_model.save", "rb"))
    scaler          = pickle.load(open("scaler.pkl",                "rb"))
    feature_columns = pickle.load(open("feature_columns.pkl",       "rb"))
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

# ── Styles ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%); }
.title {
    text-align: center; font-size: 44px; font-weight: 900;
    background: linear-gradient(90deg, #ff4b4b, #ff7b7b);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.subtitle { text-align: center; font-size: 18px; color: #555; margin-bottom: 25px; }
.card {
    background: rgba(255,255,255,0.75); backdrop-filter: blur(10px);
    padding: 25px; border-radius: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08); margin-bottom: 20px;
}
.result { padding: 25px; border-radius: 16px; text-align: center; font-size: 22px; font-weight: 700; }
.stButton>button {
    width: 100%; padding: 12px; font-size: 18px; font-weight: bold;
    border-radius: 12px; background: linear-gradient(90deg, #ff4b4b, #ff7676);
    color: white; border: none;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "❤️ Prediction", "📊 Dataset", "👨‍⚕️ About"])

# ════════════════════════════════════════════════════════════════
# 🏠 HOME
# ════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown(
        '<div class="title">❤️ Heart Disease AI System</div>'
        '<div class="subtitle">AI-powered medical risk analysis for early detection</div>',
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(
            "https://images.unsplash.com/photo-1579684385127-1ef15d508118"
            "?auto=format&fit=crop&w=1200&q=80", width=700
        )
    st.info("👉 Use the sidebar to navigate to **❤️ Prediction** and start your analysis.")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# ❤️ PREDICTION
# ════════════════════════════════════════════════════════════════

elif page == "❤️ Prediction":

    st.markdown("""
    <h1 style='text-align:center;color:#e63946;'>
        ❤️ Heart Disease Risk Prediction
    </h1>
    <p style='text-align:center;color:gray;font-size:18px;'>
        Enter patient health details below for AI-based risk assessment
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3, gap="large")

    # Column 1
    with col1:
        st.subheader("👤 Personal Details")

        age = st.slider("Age", 1, 100, 50)

        sex = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        cp = st.selectbox(
            "Chest Pain Type",
            [
                "Typical Angina",
                "Atypical Angina",
                "Non-anginal Pain",
                "Asymptomatic"
            ]
        )

        fbs = st.radio(
            "Fasting Blood Sugar >120",
            [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )

    # Column 2
    with col2:
        st.subheader("🩺 Vital Signs")

        trestbps = st.slider(
            "Resting Blood Pressure",
            80, 250, 120
        )

        chol = st.slider(
            "Cholesterol",
            100, 600, 200
        )

        thalach = st.slider(
            "Maximum Heart Rate",
            60, 250, 150
        )

        oldpeak = st.slider(
            "Oldpeak (ST Depression)",
            0.0, 10.0, 1.0, 0.1
        )

    # Column 3
    with col3:
        st.subheader("📈 Cardiac Tests")

        restecg = st.selectbox(
            "Rest ECG",
            [0, 1, 2]
        )

        exang = st.radio(
            "Exercise Induced Angina",
            [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )

        slope = st.selectbox(
            "Slope",
            [0, 1, 2]
        )

        ca = st.selectbox(
            "Major Vessels",
            [0, 1, 2, 3]
        )

        thal = st.selectbox(
            "Thalassemia",
            [0, 1, 2, 3]
        )

    st.markdown("<br>", unsafe_allow_html=True)

    center1, center2, center3 = st.columns([1,2,1])

    with center2:
        predict_btn = st.button(
            "🔍 Analyze Heart Risk",
            use_container_width=True
        )

    if predict_btn:

        sex_val = 1 if sex == "Male" else 0

        cp_val = {
            "Typical Angina": 0,
            "Atypical Angina": 1,
            "Non-anginal Pain": 2,
            "Asymptomatic": 3
        }[cp]

        input_dict = {}

        for col in feature_columns:
            mapping = {
                "age": age,
                "sex": sex_val,
                "chest_pain_type": cp_val,
                "resting_blood_pressure": trestbps,
                "cholesterol": chol,
                "fasting_blood_sugar": fbs,
                "rest_ecg": restecg,
                "Max_heart_rate": thalach,
                "exercise_induced_angina": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "vessels_colored_by_flourosopy": ca,
                "thalassemia": thal
            }

            input_dict[col] = mapping.get(col, 0)

        df_input = pd.DataFrame([input_dict])
        df_scaled = scaler.transform(df_input)

        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0]
        risk_percent = int(probability[1] * 100)

        st.markdown("---")

        st.markdown(
            "<h2 style='text-align:center;'>📊 Prediction Result</h2>",
            unsafe_allow_html=True
        )

        if prediction == 1:

            st.error(
                f"⚠️ High Risk Detected ({risk_percent}% Risk)"
            )

            st.progress(risk_percent)

        else:

            st.success(
                f"✅ Low Risk Detected ({risk_percent}% Risk)"
            )

            st.progress(risk_percent)

        with st.expander("📋 Patient Input Summary"):
            st.dataframe(
                df_input.T.rename(columns={0: "Value"}),
                use_container_width=True
            )
    st.markdown("""
<style>
div[data-baseweb="select"] > div {
    background-color: #0F172A;
    color: #E5E7EB;
    border: 2px solid #2563EB;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)
# ════════════════════════════════════════════════════════════════
# 📊 DATASET
# ════════════════════════════════════════════════════════════════
elif page == "📊 Dataset":
    st.markdown("## 📊 Dataset Overview")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
| # | Feature | Description |
|---|---------|-------------|
| 1 | Age | Patient age in years |
| 2 | Sex | 1=Male, 0=Female |
| 3 | Chest Pain Type | 0=Typical Angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic |
| 4 | Resting Blood Pressure | mm Hg on admission |
| 5 | Cholesterol | Serum cholesterol in mg/dl |
| 6 | Fasting Blood Sugar | 1 if > 120 mg/dl, else 0 |
| 7 | Rest ECG | 0/1/2 |
| 8 | Max Heart Rate | Maximum heart rate achieved |
| 9 | Exercise Angina | 1=Yes, 0=No |
| 10 | Oldpeak | ST depression induced by exercise |
| 11 | Slope | Slope of peak exercise ST segment |
| 12 | Major Vessels | Vessels colored by fluoroscopy (0–3) |
| 13 | Thalassemia | 0–3 |
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"**Training feature count:** `{len(feature_columns)}`")
    st.code(str(feature_columns))
    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# 👨‍⚕️ ABOUT
# ════════════════════════════════════════════════════════════════

elif page == "👨‍⚕️ About":

    st.markdown("""
    <h1 style='text-align:center;color:#e63946;'>
        👨‍⚕️ About
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### ❤️ Heart Disease Prediction System

    This Machine Learning application predicts the likelihood of heart disease
    using patient health parameters and an AdaBoost Classifier model.

    The system analyzes 13 clinical features and provides a risk prediction
    to support early awareness and preventive healthcare.
    """)

    st.markdown("### 🛠️ Technologies Used")
    st.write("Python • Streamlit • Scikit-Learn • Pandas • NumPy")

    st.markdown("### ⚠️ Disclaimer")
    st.warning(
        "This tool is for educational purposes only and should not be used as a substitute for professional medical advice."
    )

    st.markdown("---")

    st.markdown(
        "<center>❤️ Developed using Machine Learning & Streamlit</center>",
        unsafe_allow_html=True
    )

