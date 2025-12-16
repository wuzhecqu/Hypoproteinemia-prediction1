import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings

# ========== 临时注释SHAP/Matplotlib（避免导入报错） ==========
# import shap
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import seaborn as sns

warnings.filterwarnings('ignore')

# ===================== 0. Global Configuration (English) =====================
st.set_page_config(
    page_title="Hypoproteinemia Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== 1. Path Configuration =====================
MODEL_PATH = "lgb_model_weights.pkl"
VAL_DATA_PATH = "validation_data.xlsx"

# ===================== 2. Model Loading =====================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found! Path: {MODEL_PATH}")
        st.stop()

    try:
        with open(MODEL_PATH, 'rb') as f:
            model_metadata = pickle.load(f)
        
        model = model_metadata.get('model')
        imputer = model_metadata.get('imputer')
        scaler = model_metadata.get('scaler')
        feature_cols = model_metadata.get('feature_cols')
        feature_descs = model_metadata.get('feature_descriptions', {})
        target_mapping = model_metadata.get('target_mapping', {0: 'No Hypoproteinemia', 1: 'Hypoproteinemia'})

        if model is None or imputer is None or scaler is None or feature_cols is None:
            st.error("❌ Model corrupted! Missing core components")
            st.stop()

        # ========== 临时注释SHAP相关 ==========
        # explainer = shap.TreeExplainer(model)
        explainer = None
        shap_values_val = None
        X_val = None
        y_val = None
        val_df = pd.DataFrame(columns=feature_cols)

        return model, imputer, scaler, feature_cols, feature_descs, target_mapping, explainer, shap_values_val, X_val, y_val, val_df
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

model, imputer, scaler, feature_cols, feature_descs, target_mapping, explainer, shap_values_val, X_val, y_val, val_df = load_model()

# ===================== 3. Sidebar: Only Single Sample Prediction =====================
st.sidebar.title("Function Menu")
function_choice = st.sidebar.radio(
    "Select Function",
    ["🔮 Single Sample Prediction"]  # 临时只保留单样本预测
)

# ===================== 4. Single Sample Prediction (Core Only) =====================
if function_choice == "🔮 Single Sample Prediction":
    st.title("Postoperative Hypoproteinemia - Single Sample Prediction")
    st.markdown("### Enter Patient Clinical Features")

    # Input Form
    input_data = {}
    col1, col2 = st.columns(2)
    numeric_cols = feature_cols

    with col1:
        st.subheader("Clinical Features (1)")
        for feat in numeric_cols[:len(numeric_cols)//2]:
            min_val = float(val_df[feat].min()) if not val_df[feat].isna().all() else 0.0
            max_val = float(val_df[feat].max()) if not val_df[feat].isna().all() else 100.0
            mean_val = float(val_df[feat].median()) if not val_df[feat].isna().all() else 50.0
            input_data[feat] = st.number_input(f"{feat}\n({feature_descs[feat]})", min_val, max_val, mean_val, 0.1)

    with col2:
        st.subheader("Clinical Features (2)")
        for feat in numeric_cols[len(numeric_cols)//2:]:
            min_val = float(val_df[feat].min()) if not val_df[feat].isna().all() else 0.0
            max_val = float(val_df[feat].max()) if not val_df[feat].isna().all() else 100.0
            mean_val = float(val_df[feat].median()) if not val_df[feat].isna().all() else 50.0
            input_data[feat] = st.number_input(f"{feat}\n({feature_descs[feat]})", min_val, max_val, mean_val, 0.1)

    # Prediction Button
    if st.button("🚀 Start Prediction", type="primary"):
        input_df = pd.DataFrame([input_data])
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)

        # Prediction
        pred_proba = model.predict_proba(input_scaled)[0, 1]
        pred_label = 1 if pred_proba >= 0.5 else 0
        pred_text = target_mapping[pred_label]

        # Show Result
        st.markdown("### Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Class", pred_text)
        with col2:
            st.metric("Hypoproteinemia Risk Probability", f"{pred_proba:.2%}")

# ===================== 5. Footer =====================
st.markdown("---")
st.markdown("© 2025 Hypoproteinemia Prediction Model | Streamlit Web App")
