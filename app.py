import streamlit as st
import pandas as pd
import numpy as np
import pickle  # 适配pickle保存的模型（替代joblib）
import os
import warnings
import shap
import matplotlib
# 核心修复：兼容Streamlit Cloud无GUI环境
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ===================== 0. Global Configuration (English) =====================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# Page Configuration
st.set_page_config(
    page_title="Hypoproteinemia Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== 1. Path Configuration (Cloud-Compatible) =====================
# 模型权重文件（你的LightGBM模型：lgb_model_weights.pkl）
MODEL_PATH = "lgb_model_weights.pkl"
# 验证集文件（用于SHAP全局分析）
VAL_DATA_PATH = "validation_data.xlsx"

# ===================== 2. Model Loading (适配pickle保存的LightGBM模型) =====================
@st.cache_resource
def load_model():
    """Load LightGBM model (pickle format) with validation"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found! Path: {MODEL_PATH}")
        st.stop()

    try:
        # 加载你训练脚本保存的model_metadata（pickle格式）
        with open(MODEL_PATH, 'rb') as f:
            model_metadata = pickle.load(f)
        
        # 提取核心组件（匹配你训练脚本的model_metadata结构）
        model = model_metadata.get('model')
        imputer = model_metadata.get('imputer')
        scaler = model_metadata.get('scaler')
        feature_cols = model_metadata.get('feature_cols')
        feature_descs = model_metadata.get('feature_descriptions', {})
        target_mapping = model_metadata.get('target_mapping', {0: 'No Hypoproteinemia', 1: 'Hypoproteinemia'})

        # 验证核心组件
        if model is None or imputer is None or scaler is None or feature_cols is None:
            st.error("❌ Model corrupted! Missing core components (model/imputer/scaler/feature_cols)")
            st.stop()

        # 预生成SHAP解释器（基于验证集）
        # 加载验证集用于SHAP全局分析
        val_df = pd.read_excel(VAL_DATA_PATH, header=0, engine='openpyxl')
        val_df.columns = [col.strip() for col in val_df.columns]
        X_val_raw = val_df[feature_cols]
        
        # 预处理验证集
        X_val_imputed = imputer.transform(X_val_raw)
        X_val = scaler.transform(X_val_imputed)
        y_val = val_df['Hypoproteinemia'].map({1: 1, 2: 0})  # 匹配你的目标变量映射
        
        # 初始化SHAP解释器
        explainer = shap.TreeExplainer(model)
        shap_values_val = explainer.shap_values(X_val)

        return model, imputer, scaler, feature_cols, feature_descs, target_mapping, explainer, shap_values_val, X_val, y_val, val_df
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

# Load all model components
model, imputer, scaler, feature_cols, feature_descs, target_mapping, explainer, shap_values_val, X_val, y_val, val_df = load_model()

# ===================== 3. Sidebar: Function Selection (Only 2 Options) =====================
st.sidebar.title("Function Menu")
function_choice = st.sidebar.radio(
    "Select Function",
    ["🔮 Single Sample Prediction", "📈 Interpretability Analysis"]
)

# ===================== 4. Single Sample Prediction (Core Function) =====================
if function_choice == "🔮 Single Sample Prediction":
    st.title("Postoperative Hypoproteinemia - Single Sample Prediction")
    st.markdown("### Enter Patient Clinical Features")

    # Build input form (dynamic based on feature_cols)
    input_data = {}
    col1, col2 = st.columns(2)

    # Split features into numeric (all features are numeric for your LightGBM model)
    numeric_cols = feature_cols  # Your model uses numeric features only
    with col1:
        st.subheader("Clinical Numeric Features")
        for feat in numeric_cols[:len(numeric_cols)//2]:
            # Get stats from validation data for input range
            min_val = float(val_df[feat].min()) if not val_df[feat].isna().all() else 0.0
            max_val = float(val_df[feat].max()) if not val_df[feat].isna().all() else 100.0
            mean_val = float(val_df[feat].median()) if not val_df[feat].isna().all() else 50.0

            input_data[feat] = st.number_input(
                f"{feat}\n({feature_descs[feat]})",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=0.1
            )

    with col2:
        st.subheader("Clinical Numeric Features (Cont.)")
        for feat in numeric_cols[len(numeric_cols)//2:]:
            min_val = float(val_df[feat].min()) if not val_df[feat].isna().all() else 0.0
            max_val = float(val_df[feat].max()) if not val_df[feat].isna().all() else 100.0
            mean_val = float(val_df[feat].median()) if not val_df[feat].isna().all() else 50.0

            input_data[feat] = st.number_input(
                f"{feat}\n({feature_descs[feat]})",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=0.1
            )

    # Prediction button
    if st.button("🚀 Start Prediction", type="primary"):
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])

        # Preprocess input (match training pipeline)
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)

        # Prediction
        pred_proba = model.predict_proba(input_scaled)[0, 1]
        pred_label = 1 if pred_proba >= 0.5 else 0
        pred_text = target_mapping[pred_label]

        # Show results
        st.markdown("### Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Class", pred_text)
            st.metric("Hypoproteinemia Risk Probability", f"{pred_proba:.2%}")
        with col2:
            # SHAP Single Sample Explanation
            st.markdown("#### Feature Impact Explanation (SHAP Force Plot)")
            shap.initjs()
            shap_val = explainer.shap_values(input_scaled)
            # JS-based Force Plot (compatible with cloud)
            force_plot = shap.force_plot(
                explainer.expected_value,
                shap_val[0],
                input_data,
                feature_names=feature_cols,
                matplotlib=False,
                show=False
            )
            st.components.v1.html(shap.getjs() + force_plot.html(), height=120)

# ===================== 5. Interpretability Analysis (SHAP) =====================
elif function_choice == "📈 Interpretability Analysis":
    st.title("Model Interpretability Analysis (SHAP)")

    tab1, tab2, tab3 = st.tabs(["📊 SHAP Summary Plot", "🔍 Feature Dependence Plot", "🧬 Single Sample Explanation"])

    # Tab1: SHAP Summary Plot
    with tab1:
        st.markdown("### SHAP Summary Plot (Validation Set)")
        st.markdown("""
        - Y-axis: Feature importance (top = more impactful on prediction)
        - X-axis: SHAP value (positive = increase hypoproteinemia risk, negative = decrease risk)
        - Color: Feature value (red = high value, blue = low value)
        """)
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            shap_values_val,
            X_val,
            feature_names=feature_cols,
            plot_type="dot",
            show=False,
            ax=ax,
            cmap=plt.get_cmap("coolwarm")
        )
        st.pyplot(fig)

    # Tab2: Feature Dependence Plot
    with tab2:
        st.markdown("### Feature Dependence Plot (Top 5 Features)")
        # Calculate top 5 features by SHAP importance
        shap_importance = np.abs(shap_values_val).mean(axis=0)
        top5_feat = [feature_cols[i] for i in np.argsort(shap_importance)[-5:]][::-1]
        selected_feat = st.selectbox("Select Feature to Analyze", top5_feat)

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.dependence_plot(
            selected_feat,
            shap_values_val,
            val_df[feature_cols],
            feature_names=feature_cols,
            show=False,
            ax=ax,
            alpha=0.6,
            dot_size=20
        )
        ax.set_title(f"Feature Dependence Plot - {selected_feat}")
        st.pyplot(fig)

    # Tab3: Single Sample Explanation (Validation Set)
    with tab3:
        st.markdown("### Single Sample SHAP Explanation (Validation Set)")
        # Select sample index
        sample_idx = st.slider("Select Validation Sample Index", 0, len(X_val) - 1, 100)
        # Show sample features
        st.markdown("#### Sample Clinical Features")
        sample_data = val_df.iloc[sample_idx][feature_cols]
        st.write(sample_data)

        # Show SHAP Force Plot (Matplotlib version)
        st.markdown("#### Feature Impact on Prediction")
        fig, ax = plt.subplots(figsize=(12, 4))
        shap.force_plot(
            explainer.expected_value,
            shap_values_val[sample_idx],
            val_df.iloc[sample_idx][feature_cols],
            feature_names=feature_cols,
            matplotlib=True,
            show=False,
            figsize=(12, 4),
            ax=ax
        )
        true_label = "Hypoproteinemia" if y_val.iloc[sample_idx] == 1 else "No Hypoproteinemia"
        ax.set_title(f"Sample {sample_idx} - True Label: {true_label}")
        st.pyplot(fig)

# ===================== 6. Footer =====================
st.markdown("---")
st.markdown("© 2025 Hypoproteinemia Prediction Model | Streamlit Web App | LightGBM + SHAP Interpretability | Research Use Only")
