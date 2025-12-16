import streamlit as st
import numpy as np
import pickle
import os
import warnings

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

# ===================== 2. Model Loading (无pandas) =====================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found! Path: {MODEL_PATH}")
        st.stop()

    try:
        with open(MODEL_PATH, 'rb') as f:
            model_metadata = pickle.load(f)
        
        # 提取核心组件（匹配你的训练脚本）
        model = model_metadata.get('model')
        imputer = model_metadata.get('imputer')
        scaler = model_metadata.get('scaler')
        feature_cols = model_metadata.get('feature_cols')
        # 优先从模型文件读取特征描述（100%匹配，避免手动写错）
        feature_descs = model_metadata.get('feature_descriptions', {})
        target_mapping = model_metadata.get('target_mapping', {0: 'No Hypoproteinemia', 1: 'Hypoproteinemia'})

        # 验证核心组件
        if model is None or imputer is None or scaler is None or feature_cols is None:
            st.error("❌ Model corrupted! Missing core components (model/imputer/scaler/feature_cols)")
            st.stop()

        # 🔥 自动适配所有特征，彻底避免KeyError
        feature_ranges = {}
        for feat in feature_cols:
            # 给所有特征设置默认范围（无需手动写）
            feature_ranges[feat] = (0.0, 100.0, 50.0)
            # 如果模型文件里没有该特征的描述，自动生成
            if feat not in feature_descs:
                feature_descs[feat] = f"{feat} (Clinical Feature)"

        return model, imputer, scaler, feature_cols, feature_descs, target_mapping, feature_ranges
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()
        # 硬编码特征范围（替代pandas读取验证集，避免pandas依赖）
        # 你可以根据训练数据的特征范围手动填写，示例：
        feature_ranges = {}
        for feat in feature_cols:
            # 示例：根据你的训练数据，填写每个特征的min/max/median
            # 比如 Age: min=18, max=80, median=50；Surgery_time: min=30, max=300, median=120
            # 替换为你实际的特征范围（从本地Python 3.8环境中查）
            if feat == "Age":
                feature_ranges[feat] = (18.0, 80.0, 50.0)
            elif feat == "Surgery.time":
                feature_ranges[feat] = (30.0, 300.0, 120.0)
            elif feat == "BMI":
                feature_ranges[feat] = (18.0, 35.0, 24.0)
            else:
                # 通用默认值（可根据你的特征调整）
                feature_ranges[feat] = (0.0, 100.0, 50.0)

        return model, imputer, scaler, feature_cols, feature_descs, target_mapping, feature_ranges
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

# 加载模型
model, imputer, scaler, feature_cols, feature_descs, target_mapping, feature_ranges = load_model()

# ===================== 3. Sidebar =====================
st.sidebar.title("Function Menu")
function_choice = st.sidebar.radio(
    "Select Function",
    ["🔮 Single Sample Prediction"]
)

# ===================== 4. Single Sample Prediction (无pandas) =====================
if function_choice == "🔮 Single Sample Prediction":
    st.title("Postoperative Hypoproteinemia - Single Sample Prediction")
    st.markdown("### Enter Patient Clinical Features")

    # 构造输入表单（无pandas）
    input_data = {}
    col1, col2 = st.columns(2)

    # 拆分特征显示
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
        # 构造输入数组（无pandas，用numpy）
        input_array = np.array([[input_data[feat] for feat in feature_cols]])
        
        # 预处理（和训练时一致）
        input_imputed = imputer.transform(input_array)
        input_scaled = scaler.transform(input_imputed)
        
        # 预测
        pred_proba = model.predict_proba(input_scaled)[0, 1]
        pred_label = 1 if pred_proba >= 0.5 else 0
        pred_text = target_mapping[pred_label]

        # 显示结果
        st.markdown("### Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Class", pred_text)
        with col2:
            st.metric("Hypoproteinemia Risk Probability", f"{pred_proba:.2%}")

# ===================== 5. Footer =====================
st.markdown("---")
st.markdown("© 2025 Hypoproteinemia Prediction Model | Streamlit Web App")



