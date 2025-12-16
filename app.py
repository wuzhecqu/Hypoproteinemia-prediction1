import streamlit as st
import numpy as np
import pickle
import os
import warnings
import shap  # 新增：导入SHAP

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

# ===================== 2. Model Loading (添加SHAP解释器) =====================
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

        # 新增：初始化SHAP解释器（适配LightGBM模型）
        explainer = shap.TreeExplainer(model)

        return model, imputer, scaler, feature_cols, feature_descs, target_mapping, feature_ranges, explainer
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

# 加载模型 + SHAP解释器
model, imputer, scaler, feature_cols, feature_descs, target_mapping, feature_ranges, explainer = load_model()

# ===================== 3. Sidebar =====================
st.sidebar.title("Function Menu")
function_choice = st.sidebar.radio(
    "Select Function",
    ["🔮 Single Sample Prediction", "📊 Feature Importance (SHAP)"]  # 新增：SHAP特征重要性选项
)

# ===================== 4. Single Sample Prediction (添加SHAP可解释性) =====================
if function_choice == "🔮 Single Sample Prediction":
    st.title("Postoperative Hypoproteinemia - Single Sample Prediction")
    st.markdown("### Enter Patient Clinical Features")

    # 构造输入表单
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
        # 构造输入数组（numpy）
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

        # 新增：SHAP可解释性分析（Force Plot，JS渲染，无需matplotlib）
        st.markdown("### 🧠 Model Interpretability (SHAP Force Plot)")
        # 计算SHAP值（输入为缩放后的数据）
        shap_values = explainer.shap_values(input_scaled)
        # 处理二分类模型的SHAP值（取正类的SHAP值）
        shap_values_pos = shap_values[1] if isinstance(shap_values, list) else shap_values
        
        # 渲染SHAP Force Plot（Streamlit兼容）
        shap_html = shap.force_plot(
            explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            shap_values_pos[0],
            feature_names=feature_cols,
            out_names="Hypoproteinemia Risk",
            show=False,
            matplotlib=False
        )
        # 将SHAP HTML嵌入Streamlit
        st.components.v1.html(shap_html.html(), height=300)

# ===================== 5. SHAP特征重要性（新增功能） =====================
elif function_choice == "📊 Feature Importance (SHAP)":
    st.title("Model Interpretability - SHAP Feature Importance")
    st.markdown("### Global Feature Importance (Mean Absolute SHAP Value)")

    # 生成示例数据（或加载训练集的缩放后数据，这里用随机数据演示）
    # 若有训练集，可替换为真实数据：X_train_scaled = scaler.transform(imputer.transform(X_train))
    np.random.seed(42)
    sample_data = np.random.rand(100, len(feature_cols))  # 随机生成100个样本
    sample_data_scaled = scaler.transform(sample_data)  # 缩放
    
    # 计算SHAP值
    shap_values = explainer.shap_values(sample_data_scaled)
    shap_values_pos = shap_values[1] if isinstance(shap_values, list) else shap_values
    
    # 计算特征重要性（平均绝对SHAP值）
    shap_importance = np.abs(shap_values_pos).mean(axis=0)
    importance_df = np.column_stack((feature_cols, shap_importance))
    importance_df = importance_df[np.argsort(importance_df[:, 1])[::-1]]  # 降序排序

    # 显示特征重要性表格
    st.dataframe(
        pd.DataFrame(importance_df, columns=["Feature", "SHAP Importance"]).astype({"SHAP Importance": float}),
        use_container_width=True
    )

    # 渲染SHAP Summary Plot（JS版本）
    st.markdown("### SHAP Summary Plot (Feature Impact on Prediction)")
    shap_summary = shap.summary_plot(
        shap_values_pos,
        sample_data_scaled,
        feature_names=feature_cols,
        show=False,
        plot_type="dot"
    )
    st.pyplot(shap_summary)

# ===================== 6. Footer =====================
st.markdown("---")
st.markdown("© 2025 Hypoproteinemia Prediction Model | Streamlit Web App")
