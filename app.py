import os
import subprocess
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

ARTIFACTS_DIR = "artifacts"

COLOR_BACKGROUND = "#8DA28C"
COLOR_BOX_TINT = "#EEE9C1C3"
COLOR_BOX_BORDER = "#3A230A"
COLOR_TITLE_DESC = "#EEE9C1"
COLOR_BOX_TEXT = "#A4530C"
COLOR_INPUT_BG = "#EEE9C1"
COLOR_INPUT_TEXT = "#3A230A"
COLOR_BUTTON_BG = "#3A230A"
COLOR_BUTTON_TEXT = "#EEE9C1"
COLOR_HOVER = "#8DA28C"

TARGET_LABELS = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}

FLOAT_NUMERIC_COLUMNS = {
    "Admission grade",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)",
    "Unemployment rate",
    "GDP",
}

FRIENDLY_BOOL_LABELS = {"1": "Yes", "0": "No"}

@st.cache_resource
def load_artifacts():

    # Train the model automatically if artifacts do not exist
    if not os.path.exists(f"{ARTIFACTS_DIR}/model.keras"):
        st.info("Model artifacts not found. Training the model...")

        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        subprocess.run(
            [
                "python",
                "train_model.py",
                "--train-path",
                "train.csv",
                "--output-dir",
                ARTIFACTS_DIR,
                "--epochs",
                "100",
            ],
            check=True,
        )

    model = load_model(f"{ARTIFACTS_DIR}/model.keras")
    encoder = joblib.load(f"{ARTIFACTS_DIR}/encoder.pkl")
    scaler = joblib.load(f"{ARTIFACTS_DIR}/scaler.pkl")
    feature_columns = joblib.load(f"{ARTIFACTS_DIR}/feature_columns.pkl")
    category_options = joblib.load(f"{ARTIFACTS_DIR}/category_options.pkl")
    cat_columns = joblib.load(f"{ARTIFACTS_DIR}/cat_columns.pkl")
    numeric_columns = joblib.load(f"{ARTIFACTS_DIR}/numeric_columns.pkl")
    bool_columns = joblib.load(f"{ARTIFACTS_DIR}/bool_columns.pkl")

    return (
        model,
        encoder,
        scaler,
        feature_columns,
        category_options,
        cat_columns,
        numeric_columns,
        bool_columns,
    )

def apply_theme():
    st.markdown(
        f"""
        <style>
        header[data-testid="stHeader"], [data-testid="stHeader"], footer {{
            display: none !important;
            visibility: hidden !important;
        }}
        .main .block-container {{
            padding-top: 0.5rem !important;
        }}
        .stApp {{
            background-color: {COLOR_BACKGROUND};
        }}
        @keyframes fadeSlideUp {{
            0% {{
                opacity: 0.4;
                transform: translateY(30px);
            }}
            100% {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        h1 {{
            color: {COLOR_TITLE_DESC} !important;
            text-align: center;
            font-size: 2.6rem !important;
            margin-top: 0rem !important;
            animation: fadeSlideUp 0.8s ease-out forwards;
        }}
        .main-description {{
            text-align: center;
            color: {COLOR_TITLE_DESC} !important;
            font-size: 1.3rem !important;
            margin-bottom: 1.2rem !important;
            animation: fadeSlideUp 0.8s ease-out forwards;
        }}
        .stForm {{
            background-color: {COLOR_BOX_TINT};
            border: 1px solid {COLOR_BOX_BORDER};
            border-radius: 15px;
            padding: 2rem;
            animation: fadeSlideUp 1s ease-out forwards;
            animation-delay: 0.1s;
        }}
        .stForm h2, .stForm h3, .stForm h4 {{
            color: {COLOR_BOX_TEXT} !important;
        }}
        .stForm label, .stForm .stMarkdown p, .stForm .stMarkdown span {{
            color: {COLOR_BOX_TEXT} !important;
            font-size: 1.15rem !important;
            font-weight: 600;
            transition: color 0.3s ease;
        }}
        .stForm label:hover {{
            color: {COLOR_HOVER} !important;
        }}
        div[data-baseweb="select"] > div, 
        input, 
        .stNumberInput > div > div > input {{
            background-color: {COLOR_INPUT_BG} !important;
            color: {COLOR_INPUT_TEXT} !important;
            border: none !important;
            border-radius: 5px;
            font-size: 1rem !important;
        }}
        div[data-baseweb="select"] svg {{
            fill: {COLOR_INPUT_TEXT} !important;
            color: {COLOR_INPUT_TEXT} !important;
        }}
        div[data-baseweb="select"] span {{
            color: {COLOR_INPUT_TEXT} !important;
        }}
        .stButton > button {{
            background-color: {COLOR_BUTTON_BG} !important;
            color: {COLOR_BUTTON_TEXT} !important;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            font-size: 1.1rem;
            transition: background-color 0.3s ease, color 0.3s ease;
        }}
        .stButton > button:hover {{
            background-color: {COLOR_HOVER} !important;
            color: {COLOR_BUTTON_TEXT} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def build_input_row(cat_columns, numeric_columns, bool_columns, category_options):
    values = {}

    st.subheader("Personal and Application Information")
    col1, col2 = st.columns(2)

    with col1:
        for col in cat_columns:
            if col in bool_columns:
                continue
            options = category_options[col]
            values[col] = st.selectbox(col, options=options, key=col)

    with col2:
        values["Application order"] = st.number_input(
            "Application order", min_value=0, step=1, key="Application order"
        )
        values["Age at enrollment"] = st.number_input(
            "Age at enrollment", min_value=0, step=1, key="Age at enrollment"
        )
        values["Admission grade"] = st.number_input(
            "Admission grade", min_value=0.0, step=0.1, key="Admission grade"
        )
        for col in bool_columns:
            options = category_options[col]
            display_options = [FRIENDLY_BOOL_LABELS.get(o, o) for o in options]
            choice = st.selectbox(col, options=display_options, key=col)
            reverse_map = {v: k for k, v in FRIENDLY_BOOL_LABELS.items()}
            values[col] = int(reverse_map.get(choice, choice))

    st.subheader("Curricular Units - 1st Semester")
    sem1_cols = [c for c in numeric_columns if "1st sem" in c]
    cols = st.columns(len(sem1_cols))
    for c, col_name in zip(cols, sem1_cols):
        with c:
            if col_name in FLOAT_NUMERIC_COLUMNS:
                values[col_name] = st.number_input(
                    col_name, min_value=0.0, step=0.1, key=col_name
                )
            else:
                values[col_name] = st.number_input(
                    col_name, min_value=0, step=1, key=col_name
                )

    st.subheader("Curricular Units - 2nd Semester")
    sem2_cols = [c for c in numeric_columns if "2nd sem" in c]
    cols = st.columns(len(sem2_cols))
    for c, col_name in zip(cols, sem2_cols):
        with c:
            if col_name in FLOAT_NUMERIC_COLUMNS:
                values[col_name] = st.number_input(
                    col_name, min_value=0.0, step=0.1, key=col_name
                )
            else:
                values[col_name] = st.number_input(
                    col_name, min_value=0, step=1, key=col_name
                )

    st.subheader("Economic Indicators")
    col1, col2 = st.columns(2)
    with col1:
        values["Unemployment rate"] = st.number_input(
            "Unemployment rate", step=0.1, key="Unemployment rate"
        )
    with col2:
        values["GDP"] = st.number_input("GDP", step=0.1, key="GDP")

    return values

def predict(values, model, encoder, scaler, feature_columns, cat_columns, numeric_columns):
    row = pd.DataFrame([values])

    encoded = encoder.transform(row[cat_columns])
    ohe_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_columns))

    numeric_df = row[numeric_columns].reset_index(drop=True)
    full_row = pd.concat([numeric_df, ohe_df.reset_index(drop=True)], axis=1)
    full_row = full_row.reindex(columns=feature_columns, fill_value=0)

    scaled = scaler.transform(full_row)
    probabilities = model.predict(scaled)[0]
    predicted_class = int(np.argmax(probabilities))

    return predicted_class, probabilities

def main():
    st.set_page_config(page_title="Student Outcome Predictor", layout="wide")
    apply_theme()

    st.title("Student Academic Outcome Predictor")
    st.markdown("<p class='main-description'>Fill in the student information below to predict the expected academic outcome.</p>", unsafe_allow_html=True)

    (
        model,
        encoder,
        scaler,
        feature_columns,
        category_options,
        cat_columns,
        numeric_columns,
        bool_columns,
    ) = load_artifacts()

    with st.form("prediction_form"):
        values = build_input_row(cat_columns, numeric_columns, bool_columns, category_options)
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict")

        if submitted:
            st.markdown("---")
            predicted_class, probabilities = predict(
                values, model, encoder, scaler, feature_columns, cat_columns, numeric_columns
            )
            label = TARGET_LABELS[predicted_class]

            st.subheader("Prediction Result")
            st.markdown(f"### Predicted outcome: {label}")

            for class_index, class_label in TARGET_LABELS.items():
                st.write(f"{class_label}: {probabilities[class_index] * 100:.2f}%")
                st.progress(float(probabilities[class_index]))

if __name__ == "__main__":
    main()