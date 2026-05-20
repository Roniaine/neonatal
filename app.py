import streamlit as st
import pandas as pd
import pickle

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Neonatal Mortality Prediction System",
    page_icon="🍼",
    layout="centered"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    color: #0d6efd;
    text-align: center;
    font-weight: bold;
}

.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 10px;
    border: none;
    background-color: #0d6efd;
    color: white;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #084298;
    color: white;
}

.info-box {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #0d6efd;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD TRAINED MODEL
# =========================================================
try:
    with open("RateOfDeath.pkl", "rb") as f:
        model = pickle.load(f)

except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# =========================================================
# TITLE
# =========================================================
st.title("🍼 Neonatal Mortality Prediction System")

st.markdown("""
<div class="info-box">

This AI-powered system predicts whether a newborn baby is likely 
to survive or is at high neonatal risk within the first 28 days 
after birth using maternal and newborn clinical information.

</div>
""", unsafe_allow_html=True)

# =========================================================
# INPUT SECTION
# =========================================================
st.header("📋 Enter Maternal and Baby Information")

col1, col2 = st.columns(2)

# =========================================================
# COLUMN 1
# =========================================================
with col1:

    age = st.number_input(
        "Mother's Age",
        min_value=10,
        max_value=50,
        value=25
    )

    gravida = st.number_input(
        "Gravida",
        min_value=1,
        value=1
    )

    parity = st.number_input(
        "Parity",
        min_value=0,
        value=0
    )

    gestation_age = st.number_input(
        "Gestation Age (Weeks)",
        min_value=20,
        max_value=45,
        value=38
    )

    term_preterm = st.selectbox(
        "Term / Preterm",
        ["Term", "Preterm"]
    )

    hiv_status = st.selectbox(
        "HIV Status",
        ["Negative", "Positive"]
    )

    mode_of_delivery = st.selectbox(
        "Mode of Delivery",
        ["Normal", "C-Section"]
    )

# =========================================================
# COLUMN 2
# =========================================================
with col2:

    apgar_1min = st.slider(
        "APGAR Score at 1 Minute",
        0, 10, 7
    )

    apgar_5min = st.slider(
        "APGAR Score at 5 Minutes",
        0, 10, 8
    )

    sex_of_baby = st.selectbox(
        "Sex of Baby",
        ["Male", "Female"]
    )

    weight = st.number_input(
        "Birth Weight (grams)",
        min_value=500,
        max_value=6000,
        value=3000
    )

    pnc_6hrs = st.selectbox(
        "PNC at 6 Hours",
        ["Yes", "No"]
    )

    pnc_24hrs = st.selectbox(
        "PNC at 24 Hours",
        ["Yes", "No"]
    )

# =========================================================
# FUNCTION TO ENCODE INPUTS
# =========================================================
def encode_input():

    delivery_map = {
        "Normal": 0,
        "C-Section": 1
    }

    data = pd.DataFrame({

        "AGE": [age],

        "GRAVIDA": [gravida],

        "PARITY": [parity],

        "GESTATION AGE": [gestation_age],

        "TERM/PRETERM": [
            1 if term_preterm == "Term" else 0
        ],

        "HIV STATUS": [
            1 if hiv_status == "Positive" else 0
        ],

        "MODE OF DELIVERY": [
            delivery_map[mode_of_delivery]
        ],

        "AGPAR SCORE(1MIN)": [apgar_1min],

        "AGPAR SCORE(5MIN)": [apgar_5min],

        "SEX OF BABY": [
            1 if sex_of_baby == "Male" else 0
        ],

        "WEIGHT": [
            weight / 1000
        ],

        "PNC AT 6 HRS": [
            1 if pnc_6hrs == "Yes" else 0
        ],

        "PNC AT 24 HRS ": [
            1 if pnc_24hrs == "Yes" else 0
        ]

    })

    return data

# =========================================================
# PREDICTION BUTTON
# =========================================================
if st.button("🔍 Predict Survival Outcome"):

    try:

        # Encode input
        input_data = encode_input()

        # Predict class
        prediction = model.predict(input_data)[0]

        # Predict probabilities
        probabilities = model.predict_proba(input_data)[0]

        # Convert probabilities to percentages
        survival_chance = probabilities[0] * 100
        death_risk = probabilities[1] * 100

        st.divider()

        # =================================================
        # SURVIVAL RESULT
        # =================================================
        if prediction == 0:

            st.success("✅ Prediction: Baby is likely to survive")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Survival Chance",
                    f"{survival_chance:.1f}%"
                )

            with col2:
                st.metric(
                    "Death Risk",
                    f"{death_risk:.1f}%"
                )

        # =================================================
        # HIGH RISK RESULT
        # =================================================
        else:

            st.error("❌ Prediction: Baby is at high neonatal risk")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Death Risk",
                    f"{death_risk:.1f}%"
                )

            with col2:
                st.metric(
                    "Survival Chance",
                    f"{survival_chance:.1f}%"
                )

        # =================================================
        # SHOW ENTERED DATA
        # =================================================
        with st.expander("📄 View Entered Patient Information"):

            st.dataframe(input_data)

    except Exception as e:

        st.error(f"⚠️ Error during prediction: {e}")

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<hr>

<center>

AI-Powered Neonatal Mortality Prediction System  
Developed using Streamlit and Machine Learning

</center>
""", unsafe_allow_html=True)