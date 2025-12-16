import streamlit as st
import numpy as np
import pickle
import os
import warnings
import shap  # SHAP核心导入（无需matplotlib）

warnings.filterwarnings('ignore')

# ===================== 0. Global Configuration =====================
st.set_page_config(
    page_title="Hypoproteinemia Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== 1. Path Configuration =====================
MODEL_PATH = "lgb_model_weights.pkl"

# ===================== 2. Model Loading (SHAP解释器 + 无额外依赖) =====================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found! Path: {MODEL_PATH}")
        st.stop()

    try:
        with open(MODEL_PATH, 'rb') as f:
            model_metadata = pickle.load(f)
        
        # 提取核心组件
        model = model_metadata.get('model')
        imputer = model_metadata.get('imputer')
        scaler = model_metadata.get('scaler')
        feature_cols = model_metadata.get('feature_cols')
        feature_descs = model_metadata.get('feature_descriptions', {})
        target_mapping = model_metadata.get('target_mapping', {0: 'No Hypoproteinemia', 1: 'Hypoproteinemia'})

        # 验证核心组件
        if model is None or imputer is None or scaler is None or feature_cols is None:
            st.error("❌ Model corrupted! Missing core components")
            st.stop()

        # 自动适配所有特征，避免KeyError
        feature_ranges = {}
        for feat in feature_cols:
            feature_ranges[feat] = (0.0, 100.0, 50.0)
            if feat not in feature_descs:
                feature_descs[feat] = f"{feat} (Clinical Feature)"

        # 初始化SHAP解释器（仅用于Force Plot，无需matplotlib）
        explainer = shap.TreeExplainer(model)

        return model, imputer, scaler, feature_cols, feature_descs, target_mapping, feature_ranges, explainer
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

# 加载模型 + SHAP解释器
model, imputer, scaler, feature_cols, feature_descs, target_mapping, feature_ranges, explainer = load_model()

# ===================== 3. Sidebar (仅保留预测功能，移除需绘图的特征重要性) =====================
st.sidebar.title("Function Menu")
function_choice = st.sidebar.radio(
    "Select Function",
    ["🔮 Single Sample Prediction"]  # 移除需matplotlib的特征重要性选项
)

# ===================== 4. Single Sample Prediction (核心：预测 + SHAP Force Plot) =====================
if function_choice == "🔮 Single Sample Prediction":
    st.title("Postoperative Hypoproteinemia - Single Sample Prediction")
    st.markdown("### Enter Patient Clinical Features")

    # 构造输入表单（无pandas/matplotlib）
    input_data = {}
    col1, col2 = st.columns(2)
    feature_list = list(feature_cols)

    with col1:
        st.subheader("Clinical Features (1)")
        for feat in feature_list[:len(feature_list)//2]:
            min_val, max_val, median_val = feature_ranges[feat]
            input_data[feat] = st.number_input(
                f"{feat}\n({feature_descs[feat]})",
                min_value=min_val,
                max_value=max_val,
                value=median_val,
                step=0.1
            )

    with col2:
        st.subheader("Clinical Features (2)")
        for feat in feature_list[len(feature_list)//2:]:
            min_val, max_val, median_val = feature_ranges[feat]
            input_data[feat] = st.number_input(
                f"{feat}\n({feature_descs[feat]})",
                min_value=min_val,
                max_value=max_val,
                value=median_val,
                step=0.1
            )

    # 预测按钮
    if st.button("🚀 Start Prediction", type="primary"):
        # 构造输入数组（纯numpy）
        input_array = np.array([[input_data[feat] for feat in feature_cols]])
        
        # 预处理
        input_imputed = imputer.transform(input_array)
        input_scaled = scaler.transform(input_imputed)
        
        # 预测
        pred_proba = model.predict_proba(input_scaled)[0, 1]
        pred_label = 1 if pred_proba >= 0.5 else 0
        pred_text = target_mapping[pred_label]

        # 显示预测结果
        st.markdown("### 📈 Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Class", pred_text)
        with col2:
            st.metric("Hypoproteinemia Risk Probability", f"{pred_proba:.2%}")

        # 🔥 核心：SHAP Force Plot（JS渲染，无matplotlib/pandas）
        st.markdown("### 🧠 Model Interpretability (SHAP Force Plot)")
        st.info("Each feature's impact on the prediction (red=increase risk, blue=decrease risk)")
        
        # 计算SHAP值（适配LightGBM二分类模型）
        shap_values = explainer.shap_values(input_scaled)
        shap_values_pos = shap_values[1] if isinstance(shap_values, list) else shap_values
        
        # 渲染SHAP Force Plot（纯JS，无需任何绘图库）
        shap_html = shap.force_plot(
            base_value=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            shap_values=shap_values_pos[0],
            feature_names=feature_cols,
            out_names="Hypoproteinemia Risk",
            show=False,
            matplotlib=False  # 关键：禁用matplotlib，用JS渲染
        )
        
        # 嵌入到Streamlit（自适应宽度）
        st.components.v1.html(shap_html.html(), width=800, height=200)

# ===================== 5. Footer =====================
st.markdown("---")
st.markdown("© 2025 Hypoproteinemia Prediction Model | Streamlit Web App")
