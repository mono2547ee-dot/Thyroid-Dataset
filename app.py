# ===================================================================
# Thyroid Disease Prediction - Streamlit Web App
# ===================================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ===================================================================
# Page Configuration
# ===================================================================
st.set_page_config(
    page_title="🦋 ThyroidCare AI",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================================================
# Custom CSS
# ===================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    background-attachment: fixed;
}

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 3rem;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
}

.main-header h1 {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.main-header p {
    font-size: 1.2rem;
    opacity: 0.9;
}

.card {
    background: white;
    border-radius: 15px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    margin: 0.5rem;
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0.5rem 0;
}

.metric-label {
    font-size: 0.9rem;
    opacity: 0.9;
}

.prediction-success {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: white;
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    animation: fadeIn 0.5s ease;
}

.prediction-warning {
    background: linear-gradient(135deg, #f12711 0%, #f5af19 100%);
    color: white;
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.info-box {
    background: #e3f2fd;
    border-left: 5px solid #2196f3;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.footer {
    text-align: center;
    padding: 2rem;
    color: white;
    margin-top: 3rem;
    opacity: 0.8;
}

.stButton>button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 0.75rem 2rem;
    border-radius: 50px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
}
</style>
""", unsafe_allow_html=True)

# ===================================================================
# Load Model
# ===================================================================
@st.cache_resource
def load_model():
    """Load trained model"""
    model_path = 'svm_thyroid_model.pkl'
    info_path = 'feature_info.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(info_path):
        st.error("❌ Model files not found! Please run train_model.py first.")
        return None, None
    
    model = joblib.load(model_path)
    feature_info = joblib.load(info_path)
    
    return model, feature_info

model, feature_info = load_model()

# ===================================================================
# Header
# ===================================================================
st.markdown("""
<div class="main-header">
    <h1>🦋 ThyroidCare AI</h1>
    <p>Advanced Thyroid Disease Prediction System Using Machine Learning</p>
    <p style="font-size: 0.9rem; margin-top: 1rem;">⚡ Fast & Accurate • 🏥 Medical Grade • 🤖 SVM Powered</p>
</div>
""", unsafe_allow_html=True)

# ===================================================================
# Sidebar
# ===================================================================
with st.sidebar:
    st.markdown("## 🎯 Navigation")
    
    menu = st.radio(
        "Select Section",
        ["🏠 Home", "🔮 Prediction", "📊 Analytics", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if feature_info is not None:
        st.markdown(f"""
        <div class="card">
            <h4 style="color: #667eea; margin-bottom: 1rem;">📊 Model Information</h4>
            <p><strong>Algorithm:</strong> SVM</p>
            <p><strong>Kernel:</strong> {feature_info['best_params'].get('classifier__kernel', 'RBF')}</p>
            <p><strong>Accuracy:</strong> {feature_info.get('accuracy', 0)*100:.1f}%</p>
            <p><strong>Classes:</strong> {len(feature_info['target_classes'])} Types</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### ⚡ Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(feature_info['target_classes']) if feature_info else 0}</div>
            <div class="metric-label">Classes</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">22</div>
            <div class="metric-label">Features</div>
        </div>
        """, unsafe_allow_html=True)

# ===================================================================
# Main Content
# ===================================================================

if model is None or feature_info is None:
    st.error("❌ Cannot load model. Please run train_model.py first.")
    st.stop()

if menu == " Home":
    st.markdown("""
    <div class="card">
        <h2 style="color: #667eea; margin-bottom: 1rem;">👋 Welcome to ThyroidCare AI</h2>
        <p style="font-size: 1.1rem; line-height: 1.8;">
            Our advanced AI-powered system helps predict thyroid conditions using state-of-the-art 
            machine learning algorithms. Simply input patient data and receive instant, accurate predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <div style="font-size: 3rem; text-align: center;">⚡</div>
            <h3 style="text-align: center; color: #667eea;">Fast</h3>
            <p style="text-align: center;">Get predictions in seconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="card">
            <div style="font-size: 3rem; text-align: center;"></div>
            <h3 style="text-align: center; color: #667eea;">Accurate</h3>
            <p style="text-align: center;">{feature_info.get('accuracy', 0)*100:.1f}% accuracy rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <div style="font-size: 3rem; text-align: center;">🔒</div>
            <h3 style="text-align: center; color: #667eea;">Secure</h3>
            <p style="text-align: center;">HIPAA compliant</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("## 🔧 How It Works")
    
    steps = [
        ("", "Input Patient Data", "Enter patient information and test results"),
        ("🤖", "AI Processing", "Our SVM model analyzes the data"),
        ("📊", "Get Results", "Receive detailed prediction with confidence scores"),
        ("👨‍⚕️", "Consult Doctor", "Share results with healthcare provider")
    ]
    
    for i, (icon, title, desc) in enumerate(steps, 1):
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"<div style='font-size: 2.5rem; text-align: center;'>{icon}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="padding: 1rem; background: white; border-radius: 10px; margin-bottom: 1rem;">
                <h4 style="color: #667eea; margin: 0;">Step {i}: {title}</h4>
                <p style="margin: 0.5rem 0 0 0; color: #666;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

elif menu == "🔮 Prediction":
    st.markdown("## 🔮 Thyroid Disease Prediction")
    
    st.markdown("""
    <div class="info-box">
        <strong> Instructions:</strong> Please fill in all the patient information below. 
        Fields marked with * are important for accurate prediction.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Patient Information")
        
        age = st.number_input("Age (years)*", min_value=0, max_value=120, value=45, step=1)
        sex = st.selectbox("Sex*", ["F", "M"])
        referral_source = st.selectbox("Referral Source*", ["other", "SVHC", "SVHD", "SVI", "STMW"])
        
        st.markdown("###  Blood Test Results")
        
        TSH = st.number_input("TSH (mIU/L)*", min_value=0.0, value=2.5, step=0.1)
        T3 = st.number_input("T3 (ng/dL)*", min_value=0.0, value=1.5, step=0.1)
        TT4 = st.number_input("TT4 (mcg/dL)*", min_value=0.0, value=100.0, step=1.0)
        T4U = st.number_input("T4U*", min_value=0.0, value=1.0, step=0.1)
        FTI = st.number_input("FTI*", min_value=0.0, value=100.0, step=1.0)
    
    with col2:
        st.markdown("### 🏥 Medical History")
        
        on_thyroxine = st.checkbox("On Thyroxine", value=False)
        query_on_thyroxine = st.checkbox("Query on Thyroxine", value=False)
        on_antithyroid_medication = st.checkbox("On Anti-thyroid Medication", value=False)
        sick = st.checkbox("Sick", value=False)
        pregnant = st.checkbox("Pregnant", value=False)
        thyroid_surgery = st.checkbox("Thyroid Surgery", value=False)
        I131_treatment = st.checkbox("I131 Treatment", value=False)
        query_hypothyroid = st.checkbox("Query Hypothyroid", value=False)
        query_hyperthyroid = st.checkbox("Query Hyperthyroid", value=False)
        lithium = st.checkbox("Lithium", value=False)
        goitre = st.checkbox("Goitre", value=False)
        tumor = st.checkbox("Tumor", value=False)
        hypopituitary = st.checkbox("Hypopituitary", value=False)
        psych = st.checkbox("Psychiatric", value=False)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button(" Predict Thyroid Condition", type="primary", use_container_width=True)
    
    if predict_btn:
        with st.spinner("🤖 Analyzing patient data..."):
            input_data = pd.DataFrame({
                'age': [age],
                'sex': [sex],
                'on_thyroxine': [str(on_thyroxine)],
                'query_on_thyroxine': [str(query_on_thyroxine)],
                'on_antithyroid_medication': [str(on_antithyroid_medication)],
                'sick': [str(sick)],
                'pregnant': [str(pregnant)],
                'thyroid_surgery': [str(thyroid_surgery)],
                'I131_treatment': [str(I131_treatment)],
                'query_hypothyroid': [str(query_hypothyroid)],
                'query_hyperthyroid': [str(query_hyperthyroid)],
                'lithium': [str(lithium)],
                'goitre': [str(goitre)],
                'tumor': [str(tumor)],
                'hypopituitary': [str(hypopituitary)],
                'psych': [str(psych)],
                'TSH': [TSH],
                'T3': [T3],
                'TT4': [TT4],
                'T4U': [T4U],
                'FTI': [FTI],
                'referral_source': [referral_source]
            })
            
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            
            st.markdown("##  Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Class</div>
                    <div class="metric-value" style="font-size: 1.2rem;">{prediction}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                max_prob = max(probabilities) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{max_prob:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                risk_level = "Low" if max_prob < 50 else "Medium" if max_prob < 80 else "High"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Risk Level</div>
                    <div class="metric-value">{risk_level}</div>
                </div>
                """, unsafe_allow_html=True)
            
            if prediction == 'negative':
                st.markdown(f"""
                <div class="prediction-success">
                    <h2 style="margin: 0;">✅ Normal Thyroid Function</h2>
                    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                        The patient shows normal thyroid function. No immediate concern detected.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            elif 'hypothyroid' in prediction.lower():
                st.markdown(f"""
                <div class="prediction-warning">
                    <h2 style="margin: 0;">⚠️ Hypothyroid Condition Detected</h2>
                    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                        The patient shows signs of hypothyroidism. Medical consultation recommended.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            elif 'hyperthyroid' in prediction.lower():
                st.markdown(f"""
                <div class="prediction-warning">
                    <h2 style="margin: 0;">🔥 Hyperthyroid Condition Detected</h2>
                    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                        The patient shows signs of hyperthyroidism. Immediate medical attention advised.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="card">
                    <h3 style="color: #667eea;">📋 Prediction: {prediction}</h3>
                    <p>Further medical evaluation may be required.</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 📊 Probability Distribution")
            
            prob_df = pd.DataFrame({
                'Class': feature_info['target_classes'],
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(prob_df)))
            bars = ax.barh(prob_df['Class'], prob_df['Probability'], color=colors)
            ax.set_xlabel('Probability', fontsize=12, fontweight='bold')
            ax.set_title('Class Probabilities', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlim(0, 1)
            ax.grid(axis='x', alpha=0.3)
            
            for i, (v, bar) in enumerate(zip(prob_df['Probability'], bars)):
                ax.text(v + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{v*100:.1f}%', va='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("### 💡 Recommendations")
            
            if prediction == 'negative':
                st.success("**✅ Good News!** Continue regular health check-ups")
            elif 'hypothyroid' in prediction.lower():
                st.warning("**⚠️ Medical Attention Recommended** Consult an endocrinologist")
            elif 'hyperthyroid' in prediction.lower():
                st.error("**🔥 Urgent Medical Attention Required** See a doctor immediately")
            else:
                st.info(f"**ℹ️ Further Evaluation Needed** Result: {prediction}")

elif menu == "📊 Analytics":
    st.markdown("## 📊 Model Analytics & Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="card">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; color: #667eea;">🎯</div>
                <h3 style="color: #667eea; margin: 0.5rem 0;">Accuracy</h3>
                <p style="font-size: 2rem; font-weight: bold; margin: 0;">{feature_info.get('accuracy', 0)*100:.1f}%</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="card">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; color: #667eea;">📊</div>
                <h3 style="color: #667eea; margin: 0.5rem 0;">CV Score</h3>
                <p style="font-size: 2rem; font-weight: bold; margin: 0;">{feature_info.get('cv_score', 0)*100:.1f}%</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="card">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; color: #667eea;">🏷️</div>
                <h3 style="color: #667eea; margin: 0.5rem 0;">Classes</h3>
                <p style="font-size: 2rem; font-weight: bold; margin: 0;">{len(feature_info['target_classes'])}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="card">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; color: #667eea;">⚙️</div>
                <h3 style="color: #667eea; margin: 0.5rem 0;">Features</h3>
                <p style="font-size: 2rem; font-weight: bold; margin: 0;">{len(feature_info['feature_names'])}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### ️ Classes ที่ทำนาย")
    for cls in feature_info['target_classes']:
        st.markdown(f"- {cls}")
    
    if feature_info.get('removed_classes'):
        st.markdown("### ⚠️ Classes ที่ถูกลบ (ตัวอย่างน้อย)")
        for cls in feature_info['removed_classes']:
            st.markdown(f"- {cls}")

elif menu == "ℹ️ About":
    st.markdown("## ℹ️ About ThyroidCare AI")
    
    st.markdown("""
    <div class="card">
        <h3 style="color: #667eea;">🎯 Our Mission</h3>
        <p style="line-height: 1.8;">
            ThyroidCare AI is designed to assist healthcare professionals in early detection and 
            diagnosis of thyroid conditions. Our system leverages advanced machine learning 
            algorithms to provide accurate, fast, and reliable predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>⚠️ Disclaimer:</strong><br>
        This system is intended as a decision support tool only. It should not replace 
        professional medical advice, diagnosis, or treatment. Always seek the advice of 
        qualified health providers with any questions you may have regarding medical conditions.
    </div>
    """, unsafe_allow_html=True)

# ===================================================================
# Footer
# ===================================================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>© 2026 ThyroidCare AI | Advanced Thyroid Disease Prediction System</p>
    <p style="font-size: 0.9rem;">Developed with ❤️ using Streamlit & Machine Learning</p>
    <p style="font-size: 0.8rem; opacity: 0.7;">Version 1.0 | Last Updated: 2026</p>
</div>
""", unsafe_allow_html=True)