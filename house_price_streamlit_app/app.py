import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import os

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    layout="wide"
)

st.title("House Price Prediction App")
st.markdown(
    """
    This app predicts **house sale prices** using a trained **XGBoost model**.  
    You can either:
    - Predict a **single house price**
    - Upload an **Excel file** for **bulk predictions**
    """
)

# --------------------------------------------------
# Load trained objects
# --------------------------------------------------
def load_artifacts():
    base_path = os.path.dirname(__file__)

    model = joblib.load(os.path.join(base_path, "best_xgb_model.pkl"))
    ct = joblib.load(os.path.join(base_path, "column_transformer.pkl"))
    imputer_num = joblib.load(os.path.join(base_path, "imputer_num.pkl"))
    imputer_cat = joblib.load(os.path.join(base_path, "imputer_cat.pkl"))
    features_to_drop = joblib.load(os.path.join(base_path, "features_to_drop.pkl"))
    all_features = joblib.load(os.path.join(base_path, "all_features.pkl"))

    return model, ct, imputer_num, imputer_cat, features_to_drop, all_features



model, ct, imputer_num, imputer_cat, features_to_drop, all_features = load_artifacts()

# --------------------------------------------------
# Sidebar: mode selection
# --------------------------------------------------
st.sidebar.header("Prediction Mode")

mode = st.sidebar.radio(
    "Choose prediction type:",
    ["Single House Prediction", "Bulk Prediction (Excel Upload)"]
)

# ==================================================
# SINGLE HOUSE PREDICTION
# ==================================================
if mode == "Single House Prediction":

    st.subheader("🔹 Enter House Details")

    col1, col2 = st.columns(2)

    with col1:
        OverallQual = st.slider("Overall Quality (1–10)", 1, 10, 5)
        GrLivArea = st.number_input("Above Ground Living Area (sq ft)", 300, 6000, 1500)
        TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", 0, 3000, 800)

    with col2:
        GarageCars = st.slider("Garage Capacity (Cars)", 0, 4, 2)
        YearBuilt = st.slider("Year Built", 1870, 2024, 2005)

    input_df = pd.DataFrame({
        "OverallQual": [OverallQual],
        "GrLivArea": [GrLivArea],
        "TotalBsmtSF": [TotalBsmtSF],
        "GarageCars": [GarageCars],
        "YearBuilt": [YearBuilt]
    })

    if st.button("Predict House Price"):
        # Impute numerical features
        input_df[input_df.columns] = imputer_num.transform(input_df)

        # Encode
        encoded = ct.transform(input_df)
        encoded_df = pd.DataFrame(encoded, columns=all_features)

        # Drop non-contributing features
        final_df = encoded_df.drop(columns=features_to_drop, errors="ignore")

        prediction = model.predict(final_df)[0]

        st.success(f"Estimated House Price: **${prediction:,.0f}**")


# ==================================================
# BULK PREDICTION (EXCEL UPLOAD)
# ==================================================
if mode == "Bulk Prediction (Excel Upload)":

    st.subheader("Upload Excel File for Bulk Prediction")

    uploaded_file = st.file_uploader(
        "Upload an Excel file (.xlsx)",
        type=["xlsx"]
    )

    if uploaded_file is not None:

        df = pd.read_excel(uploaded_file)

        st.write("### Preview of Uploaded Data")
        st.dataframe(df.head())

        # Handle Id column if present
        ids = None
        if "Id" in df.columns:
            ids = df["Id"]
            df = df.drop("Id", axis=1)

        # Separate numerical and categorical columns
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns

        # Impute missing values
        df[num_cols] = imputer_num.transform(df[num_cols])
        df[cat_cols] = imputer_cat.transform(df[cat_cols])

        # Encode categorical features
        encoded = ct.transform(df)
        encoded_df = pd.DataFrame(encoded, columns=all_features)

        # Drop non-contributing features
        final_df = encoded_df.drop(columns=features_to_drop, errors="ignore")

        # Predict
        predictions = model.predict(final_df)

        result_df = df.copy()
        result_df["Predicted_SalePrice"] = predictions

        if ids is not None:
            result_df.insert(0, "Id", ids)

        st.success("Prediction completed successfully!")
        st.dataframe(result_df.head())

        # Download predictions
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            result_df.to_excel(writer, index=False)

        st.download_button(
            label="📥 Download Predictions as Excel",
            data=output.getvalue(),
            file_name="house_price_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

