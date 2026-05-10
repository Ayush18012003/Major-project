"""
Depression Detection Streamlit Application
A user-friendly interface for depression detection using deep learning

Author: Data Science Team
Date: 2026-04-19
License: MIT
"""

import streamlit as st
import sys
import pandas as pd
from pathlib import Path
from utils import DepressionDetectionModel, predict_depression, comprehensive_clean_text, get_model_info

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Depression Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if 'model_obj' not in st.session_state:
    st.session_state.model_obj = None
    st.session_state.model_loaded = False
    st.session_state.load_error = None

if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# ============================================================================
# SIDEBAR - MODEL INFORMATION & SETTINGS
# ============================================================================

with st.sidebar:
    st.markdown("## 📊 Model Information")
    
    model_info = get_model_info()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Accuracy", model_info['accuracy'])
        st.metric("Precision", model_info['precision'])
        st.metric("Recall", model_info['recall'])
    
    with col2:
        st.metric("F1-Score", model_info['f1_score'])
        st.metric("AUC-ROC", model_info['auc_roc'])
        st.metric("Type", model_info['model_type'])
    
    st.markdown("---")
    
    st.markdown("## ⚙️ Settings")
    
    show_cleaning_demo = st.checkbox("Show text cleaning example", value=False)
    show_confidence_level = st.checkbox("Show confidence level", value=True)
    
    st.markdown("---")
    
    st.markdown("## ℹ️ About")
    st.info("""
    This application uses a **Bidirectional LSTM** deep learning model 
    to detect signs of depression from text input.
    
    **Performance:**
    - Trained on 20,000 balanced samples
    - Independent test accuracy: 90.20%
    - Sensitivity: 94.05% (catches depression cases)
    
    **Note:** This tool is for educational and awareness purposes. 
    It should not be used as a substitute for professional mental health evaluation.
    """)

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.markdown("# 🧠 Depression Detection AI")
st.markdown("### Detect potential signs of depression from text using Deep Learning")

# Description
st.markdown("""
This application uses a trained **Bidirectional LSTM (BiLSTM)** neural network to analyze 
text and detect potential indicators of depression. The model was trained on 20,000 text samples 
and achieves **90.20% accuracy** on independent test data.

**⚠️ Disclaimer:** This tool is for educational and awareness purposes only. 
It should **NOT** be used as a replacement for professional mental health evaluation by qualified healthcare providers.
""")

st.markdown("---")

# ============================================================================
# MODEL LOADING
# ============================================================================

if not st.session_state.model_loaded:
    st.markdown("## 🔧 Loading Model...")
    
    with st.spinner("Loading depression detection model..."):
        try:
            # Initialize model
            st.session_state.model_obj = DepressionDetectionModel()
            success, message = st.session_state.model_obj.load_model()
            
            if success:
                st.session_state.model_loaded = True
                st.success("✅ Model loaded successfully!")
            else:
                st.error(f"❌ Failed to load model: {message}")
                st.session_state.load_error = message
        
        except Exception as e:
            st.error(f"❌ Error initializing model: {str(e)}")
            st.session_state.load_error = str(e)

st.markdown("---")

# ============================================================================
# MAIN APPLICATION (Only show if model is loaded)
# ============================================================================

if st.session_state.model_loaded and st.session_state.model_obj is not None:
    
    st.markdown("## 📝 Enter Text for Analysis")
    
    # Text input
    user_input = st.text_area(
        label="Paste or type your text here:",
        placeholder="Enter text for depression detection analysis...",
        height=150,
        help="The text will be anonymously processed and analyzed by the model."
    )
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        predict_button = st.button("🔍 Analyze Text", use_container_width=True)
    
    with col2:
        clear_button = st.button("🔄 Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.prediction_result = None
        st.rerun()
    
    # ========================================================================
    # MAKE PREDICTION
    # ========================================================================
    
    if predict_button:
        if not user_input or len(user_input.strip()) == 0:
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text..."):
                result = predict_depression(st.session_state.model_obj, user_input)
                st.session_state.prediction_result = result
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    
    if st.session_state.prediction_result is not None:
        result = st.session_state.prediction_result
        
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        if result['success']:
            # Determine if depression or not
            is_depression = result['class'] == 'Depression Detected'
            
            # Display main result with color coding
            if is_depression:
                st.markdown(f"""
                <div class="warning-box">
                    <h3>⚠️ {result['class']}</h3>
                    <p>The model indicates potential signs of depression in the provided text.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-box">
                    <h3>✅ {result['class']}</h3>
                    <p>The model does not indicate significant signs of depression in the provided text.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display metrics in columns
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric(
                    "Prediction",
                    result['class'],
                    help="Classification result"
                )
            
            with metric_col2:
                probability_pct = result['probability'] * 100
                st.metric(
                    "Depression Probability",
                    f"{probability_pct:.2f}%",
                    help="Probability of depression (0-100%)"
                )
            
            with metric_col3:
                if show_confidence_level:
                    confidence_pct = result['confidence'] * 100
                    st.metric(
                        "Confidence Level",
                        f"{confidence_pct:.2f}%",
                        help="Confidence in the prediction"
                    )
            
            # Visualization: Probability bar
            st.markdown("### Confidence Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                depression_prob = result['probability']
                no_depression_prob = 1 - result['probability']
                
                st.bar_chart({
                    'Depression': depression_prob,
                    'No Depression': no_depression_prob
                })
            
            with col2:
                st.markdown(f"""
                **Probability Breakdown:**
                - 🔴 Depression: {result['probability']*100:.2f}%
                - 🟢 No Depression: {(1-result['probability'])*100:.2f}%
                """)
        else:
            # Error occurred
            st.markdown(f"""
            <div class="error-box">
                <h3>❌ Error</h3>
                <p>{result['error']}</p>
            </div>
            """, unsafe_allow_html=True)

    # ========================================================================
    # TEXT CLEANING DEMO (Optional)
    # ========================================================================
    
    if show_cleaning_demo:
        st.markdown("---")
        st.markdown("## 🧹 Text Cleaning Demo")
        st.info("This shows how the input text is cleaned before analysis")
        
        demo_text = user_input if user_input else "Check this out! 😊 Visit https://example.com @user #depression I feel so sad... don't want to do anything :("
        
        cleaned_demo = comprehensive_clean_text(demo_text)
        
        demo_col1, demo_col2 = st.columns(2)
        
        with demo_col1:
            st.markdown("**Original Text:**")
            st.write(demo_text)
        
        with demo_col2:
            st.markdown("**Cleaned Text:**")
            st.write(cleaned_demo)

else:
    if st.session_state.load_error:
        st.error(f"""
        ❌ **Model Loading Failed**
        
        Error: {st.session_state.load_error}
        
        **Troubleshooting:**
        - Ensure `depression_detection_model.keras` is in the same directory as this app
        - Ensure `tokenizer.pkl` is in the same directory as this app
        - Check that all required dependencies are installed
        """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>Depression Detection AI | Built with Streamlit & TensorFlow | 2026</p>
    <p><strong>Disclaimer:</strong> This tool is for educational purposes only and should not replace professional mental health evaluation.</p>
</div>
""", unsafe_allow_html=True)
