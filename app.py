import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

MODEL_FILE = "models/best_model.pkl"

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

# -------------------------------
# UTILS
# -------------------------------
def load_model():
    """Load model from file if it exists"""
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None
    else:
        st.warning("⚠️ No pre-trained model found. Please upload a dataset and train a model.")
        return None

def save_model(model):
    """Save trained model to file (compressed)"""
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(model, MODEL_FILE, compress=3)

def train_model(data, target_col):
    """Train Random Forest model from uploaded DataFrame"""
    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != target_col]
    num_cols = [n for n in num_cols if n != target_col]
    X = data[cat_cols + num_cols]
    y = data[target_col]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    # Reduce model size for Streamlit Cloud
    pipeline = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(
            n_estimators=50,  # Reduce number of trees
            max_depth=10,     # Limit tree depth
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    save_model(pipeline)
    return acc, cat_cols, num_cols

def make_prediction(model, input_df):
    """Single prediction with probability"""
    pred_class = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0].max()
    return pred_class, pred_proba

def batch_prediction(model, df):
    """Batch prediction with probabilities"""
    preds = model.predict(df)
    pred_probs = model.predict_proba(df).max(axis=1)
    df['PredictedClass'] = preds
    df['PredictionProbability'] = pred_probs
    return df

# -------------------------------
# MAIN APP
# -------------------------------
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns **>50K or <=50K** based on input features.")

# -------------------------------
# Upload Data & Train Model
# -------------------------------
st.sidebar.header("📂 Upload Training CSV")
train_file = st.sidebar.file_uploader("Upload CSV for Training (with target column)", type=['csv'])

if train_file is not None:
    try:
        train_df = pd.read_csv(train_file)
        st.write("### Training Data Preview", train_df.head())
        target_col = st.sidebar.selectbox("🎯 Select Target Column", train_df.columns)
        if st.sidebar.button("🚀 Train Model"):
            with st.spinner("Training model..."):
                acc, cat_cols, num_cols = train_model(train_df, target_col)
                st.success(f"✅ Model trained! Accuracy on test data: {acc:.2%}")
                st.session_state.cat_cols = cat_cols
                st.session_state.num_cols = num_cols
                st.session_state.train_df = train_df
                st.session_state.target_col = target_col
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

# -------------------------------
# Load Model
# -------------------------------
model = load_model()

if model is not None:
    st.sidebar.header("👤 Input Employee Details")
    if 'train_df' in st.session_state:
        train_df = st.session_state.train_df
        cat_cols = st.session_state.cat_cols
        num_cols = st.session_state.num_cols
        input_data = {}
        for col in cat_cols:
            options = train_df[col].dropna().unique().tolist()
            input_data[col] = st.sidebar.selectbox(col, options)
        for col in num_cols:
            min_val = int(train_df[col].min())
            max_val = int(train_df[col].max())
            default_val = int((min_val + max_val) / 2)
            input_data[col] = st.sidebar.slider(col, min_val, max_val, default_val)
        input_df = pd.DataFrame([input_data])
        st.write("### Input Data for Prediction", input_df)
        if st.button("🎯 Predict Salary Class"):
            pred, proba = make_prediction(model, input_df)
            st.success(f"Prediction: **{pred}** (Confidence: {proba:.2%})")
    else:
        st.info("📌 Upload a CSV and train a model to enable dynamic inputs.")

    # -------------------------------
    # Batch Prediction
    # -------------------------------
    st.markdown("---")
    st.header("📊 Batch Prediction")
    batch_file = st.file_uploader("Upload CSV for Batch Prediction", type=['csv'])
    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            st.write("Uploaded Batch Data Preview", batch_df.head())
            if st.button("⚡ Run Batch Prediction"):
                results_df = batch_prediction(model, batch_df)
                st.write("### Prediction Results", results_df.head())
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"❌ Error during batch prediction: {e}")
else:
    st.info("📌 To make predictions, first train a model or upload a pre-trained one.")
