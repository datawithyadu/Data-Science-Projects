import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="House Price Prediction", layout="wide")

BASE_PATH = os.path.dirname(__file__)

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_PATH, "best_xgb_model.pkl"))
    ct = joblib.load(os.path.join(BASE_PATH, "column_transformer.pkl"))
    imputer_num = joblib.load(os.path.join(BASE_PATH, "imputer_num.pkl"))
    imputer_cat = joblib.load(os.path.join(BASE_PATH, "imputer_cat.pkl"))
    features_to_drop = joblib.load(os.path.join(BASE_PATH, "features_to_drop.pkl"))
    all_features = joblib.load(os.path.join(BASE_PATH, "all_features.pkl"))
    return model, ct, imputer_num, imputer_cat, features_to_drop, all_features


model, ct, imputer_num, imputer_cat, features_to_drop, all_features = load_artifacts()

st.title("House Price Prediction")
st.write("Upload a CSV or Excel file with **all required features**")

uploaded_file = st.file_uploader(
    "Upload input file",
    type=["csv", "xlsx"]
)

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Check missing features
    missing_cols = set(all_features) - set(df.columns)
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    # Drop unwanted features
    df = df.drop(columns=features_to_drop, errors="ignore")

    # Split numeric & categorical
    num_cols = imputer_num.feature_names_in_
    cat_cols = imputer_cat.feature_names_in_

    df[num_cols] = imputer_num.transform(df[num_cols])
    df[cat_cols] = imputer_cat.transform(df[cat_cols])

    X = ct.transform(df)

    predictions = model.predict(X)

    df["Predicted_Price"] = predictions

    st.subheader("Predictions")
    st.dataframe(df[["Predicted_Price"]])

    st.download_button(
        "⬇ Download Predictions",
        df.to_csv(index=False),
        file_name="house_price_predictions.csv",
        mime="text/csv"
    )
