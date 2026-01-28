import streamlit as st
import pandas as pd
import joblib
import os
import io

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    layout="wide"
)

st.title("House Price Prediction")
st.markdown(
    """
    Upload an **Excel file** containing house features  
    to predict **Sale Prices** using a trained ML model.
    """
)

# --------------------------------------------------
# Load model artifacts safely
# --------------------------------------------------
@st.cache_resource
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
# File upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Excel file (.xlsx)",
    type=["xlsx"]
)

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Handle Id column
    ids = None
    if "Id" in df.columns:
        ids = df["Id"]
        df = df.drop("Id", axis=1)

    # Separate numeric & categorical columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    # Validate input
    if df.empty:
        st.error("Uploaded file is empty.")
        st.stop()

    # Impute missing values
    df[num_cols] = imputer_num.transform(df[num_cols])
    df[cat_cols] = imputer_cat.transform(df[cat_cols])

    # Encode categorical variables
    encoded = ct.transform(df)
    encoded_df = pd.DataFrame(encoded, columns=all_features)

    # Drop non-contributing features
    final_df = encoded_df.drop(columns=features_to_drop, errors="ignore")

    # Predict
    predictions = model.predict(final_df)

    # Prepare output
    result_df = df.copy()
    result_df["Predicted_SalePrice"] = predictions

    if ids is not None:
        result_df.insert(0, "Id", ids)

    st.success("Prediction completed successfully!")
    st.dataframe(result_df.head())

    # Download results
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result_df.to_excel(writer, index=False)

    st.download_button(
        label="Download Predictions",
        data=output.getvalue(),
        file_name="house_price_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
