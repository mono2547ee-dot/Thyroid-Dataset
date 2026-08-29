import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="🦋 ThyroidCare AI", page_icon="🦋", layout="wide")

# ===================================================================
# Custom CSS (แก้ไขให้ตัวอักษรบนพื้นขาวเป็นสีดำชัดเจน)
# ===================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
* { font-family: 'Poppins', sans-serif; }
.stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); background-attachment: fixed; }

/* --- พื้นหลังขาว/อ่อน -> บังคับตัวหนังสือสีดำ --- */
.card, .card h1, .card h2, .card h3, .card h4, .card p, .card li, .card ul,
.info-box, .info-box strong,
.stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    color: #000000 !important;
}
.card { background: white; border-radius: 15px; padding: 2rem; margin: 1rem 0; box-shadow: 0 5px 20px rgba(0,0,0,0.1); }
.info-box { background: #e3f2fd; border-left: 5px solid #2196f3; padding: 1rem; border-radius: 8px; margin: 1rem 0; }

/* --- พื้นหลังสี/Gradient -> บังคับตัวหนังสือสีขาว --- */
.main-header, .main-header h1, .main-header p,
.metric-card, .metric-card .metric-label, .metric-card .metric-value,
.prediction-success, .prediction-success h2, .prediction-success p,
.prediction-warning, .prediction-warning h2, .prediction-warning p,
.footer, .footer p {
    color: #ffffff !important;
}
.main-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 3rem; border-radius: 20px; text-align: center; margin-bottom: 2rem; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }
.main-header h1 { font-size: 3rem; font-weight: 700; margin-bottom: 0.5rem; }
.metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 12px; text-align: center; }
.metric-value { font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0; }
.prediction-success { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 2rem; border-radius: 15px; text-align: center; }
.prediction-warning { background: linear-gradient(135deg, #f12711 0%, #f5af19 100%); padding: 2rem; border-radius: 15px; text-align: center; }
.footer { text-align: center; padding: 2rem; margin-top: 3rem; opacity: 0.8; }
</style>
""", unsafe_allow_html=True)

# ===================================================================
# Smart Load or Train Function (แก้ปัญหา Version Mismatch ถาวร)
# ===================================================================
@st.cache_resource
def get_model():
    model_path = "svm_thyroid_model.pkl"
    info_path = "feature_info.pkl"
    data_path = "Thyroid-Dataset.csv"
    
    # 1. ลองโหลดโมเดลที่มีอยู่ก่อน
    if os.path.exists(model_path) and os.path.exists(info_path):
        try:
            model = joblib.load(model_path)
            info = joblib.load(info_path)
            return model, info
        except Exception:
            st.warning("⚠️ ตรวจพบโมเดลเก่าที่เข้ากันไม่ได้ กำลังเทรนโมเดลใหม่ให้เหมาะสมกับระบบ...")
            # ถ้า Error ให้ลบไฟล์เก่าทิ้งเพื่อป้องกันปัญหาซ้ำ
            if os.path.exists(model_path): os.remove(model_path)
            if os.path.exists(info_path): os.remove(info_path)

    # 2. ถ้าไม่มีไฟล์ หรือ โหลดไม่ได้ ให้เทรนใหม่จาก CSV บน Cloud
    st.info("🔄 กำลังเตรียมข้อมูลและเทรนโมเดลบน Cloud (ใช้เวลาประมาณ 10-20 วินาที)...")
    
    if not os.path.exists(data_path):
        st.error(f"❌ ไม่พบไฟล์ {data_path} กรุณาอัปโหลดไฟล์นี้ขึ้น GitHub")
        return None, None

    # โหลดและทำความสะอาดข้อมูล
    df = pd.read_csv(data_path, header=None)
    column_names = ['age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication', 
                    'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 
                    'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
                    'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'referral_source', 'status']
    
    if df.shape[1] != len(column_names):
        column_names = column_names[:df.shape[1]]
    df.columns = column_names
    df = df.replace('?', np.nan)
    
    bool_cols = ['on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication', 'sick', 'pregnant', 
                 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 
                 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych']
    for col in bool_cols:
        if col in df.columns: df[col] = df[col].astype(str)
        
    numerical_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
    for col in numerical_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df['status'] = df['status'].astype(str)
    
    # กรองคลาสที่ข้อมูลน้อยเกินไป
    class_counts = df['status'].value_counts()
    valid_classes = class_counts[class_counts >= 3].index.tolist()
    df = df[df['status'].isin(valid_classes)]
    
    X = df.drop('status', axis=1)
    y = df['status']
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessing Pipeline
    categorical_features = ['sex', 'referral_source'] + bool_cols
    numerical_features = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
    categorical_features = [c for c in categorical_features if c in X.columns]
    numerical_features = [c for c in numerical_features if c in X.columns]
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numerical_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
    ])
    
    # Train Model (แบบเร็ว)
    svm_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(random_state=42, probability=True, class_weight='balanced'))
    ])
    
    random_search = RandomizedSearchCV(
        svm_pipeline,
        {'classifier__C': [0.1, 1, 10], 'classifier__gamma': ['scale', 0.1], 'classifier__kernel': ['rbf']},
        n_iter=5, cv=2, scoring='accuracy', n_jobs=-1, random_state=42, verbose=0
    )
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    acc = accuracy_score(y_test, best_model.predict(X_test))
    
    # บันทึกโมเดลที่เทรนใหม่ (ซึ่งจะเข้ากันกับ Cloud 100%)
    joblib.dump(best_model, model_path)
    joblib.dump({
        'numerical_features': numerical_features, 'categorical_features': categorical_features,
        'bool_features': bool_cols, 'target_classes': list(best_model.classes_),
        'feature_names': X.columns.tolist(), 'best_params': random_search.best_params_,
        'accuracy': float(acc)
    }, info_path)
    
    st.success("✅ เทรนโมเดลเสร็จสิ้น! พร้อมใช้งาน")
    return best_model, joblib.load(info_path)

# เรียกใช้ฟังก์ชัน
model, feature_info = get_model()

if model is None:
    st.stop()

# ===================================================================
# UI ส่วนหัว
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
    menu = st.radio("Select Section", ["🏠 Home", "🔮 Prediction", "📊 Analytics", "ℹ️ About"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"""
    <div class="card">
        <h4 style="color: #667eea; margin-bottom: 1rem;">📊 Model Information</h4>
        <p><strong>Algorithm:</strong> SVM</p>
        <p><strong>Kernel:</strong> {feature_info['best_params'].get('classifier__kernel', 'RBF')}</p>
        <p><strong>Accuracy:</strong> {feature_info.get('accuracy', 0)*100:.1f}%</p>
        <p><strong>Classes:</strong> {len(feature_info['target_classes'])} Types</p>
    </div>
    """, unsafe_allow_html=True)

# ===================================================================
# Main Content
# ===================================================================
if menu == "🏠 Home":
    st.markdown("""<div class="card"><h2 style="color: #667eea;">👋 Welcome to ThyroidCare AI</h2><p>ระบบทำนายโรคไทรอยด์ด้วย AI ที่แม่นยำและรวดเร็ว เพียงกรอกข้อมูลผู้ป่วย ระบบจะวิเคราะห์และให้ผลลัพธ์ทันที</p></div>""", unsafe_allow_html=True)

elif menu == "🔮 Prediction":
    st.markdown("## 🔮 Thyroid Disease Prediction")
    st.markdown("""<div class="info-box"><strong>📋 Instructions:</strong> Please fill in all the patient information below.</div>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 👤 Patient Information")
        age = st.number_input("Age (years)*", min_value=0, max_value=120, value=45, step=1)
        sex = st.selectbox("Sex*", ["F", "M"])
        referral_source = st.selectbox("Referral Source*", ["other", "SVHC", "SVHD", "SVI", "STMW"])
        st.markdown("### 🩸 Blood Test Results")
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
    
    if st.button("🔮 Predict Thyroid Condition", type="primary", use_container_width=True):
        with st.spinner("🤖 Analyzing patient data..."):
            input_data = pd.DataFrame({
                'age': [age], 'sex': [sex], 'on_thyroxine': [str(on_thyroxine)], 'query_on_thyroxine': [str(query_on_thyroxine)],
                'on_antithyroid_medication': [str(on_antithyroid_medication)], 'sick': [str(sick)], 'pregnant': [str(pregnant)],
                'thyroid_surgery': [str(thyroid_surgery)], 'I131_treatment': [str(I131_treatment)],
                'query_hypothyroid': [str(query_hypothyroid)], 'query_hyperthyroid': [str(query_hyperthyroid)],
                'lithium': [str(lithium)], 'goitre': [str(goitre)], 'tumor': [str(tumor)],
                'hypopituitary': [str(hypopituitary)], 'psych': [str(psych)],
                'TSH': [TSH], 'T3': [T3], 'TT4': [TT4], 'T4U': [T4U], 'FTI': [FTI], 'referral_source': [referral_source]
            })
            
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            max_prob = max(probabilities) * 100
            
            st.markdown("## 🎯 Prediction Results")
            col1, col2, col3 = st.columns(3)
            with col1: st.markdown(f"""<div class="metric-card"><div class="metric-label">Predicted Class</div><div class="metric-value" style="font-size: 1.2rem;">{prediction}</div></div>""", unsafe_allow_html=True)
            with col2: st.markdown(f"""<div class="metric-card"><div class="metric-label">Confidence</div><div class="metric-value">{max_prob:.1f}%</div></div>""", unsafe_allow_html=True)
            with col3: 
                risk = "Low" if max_prob < 50 else "Medium" if max_prob < 80 else "High"
                st.markdown(f"""<div class="metric-card"><div class="metric-label">Risk Level</div><div class="metric-value">{risk}</div></div>""", unsafe_allow_html=True)
            
            if prediction == 'negative':
                st.markdown(f"""<div class="prediction-success"><h2 style="margin: 0;">✅ Normal Thyroid Function</h2><p>The patient shows normal thyroid function.</p></div>""", unsafe_allow_html=True)
            elif 'hypothyroid' in prediction.lower():
                st.markdown(f"""<div class="prediction-warning"><h2 style="margin: 0;">⚠️ Hypothyroid Condition Detected</h2><p>Medical consultation recommended.</p></div>""", unsafe_allow_html=True)
            elif 'hyperthyroid' in prediction.lower():
                st.markdown(f"""<div class="prediction-warning"><h2 style="margin: 0;">🔥 Hyperthyroid Condition Detected</h2><p>Immediate medical attention advised.</p></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="card"><h3 style="color: #667eea;">📋 Prediction: {prediction}</h3><p>Further medical evaluation may be required.</p></div>""", unsafe_allow_html=True)

elif menu == "📊 Analytics":
    st.markdown("## 📊 Model Analytics & Performance")
    col1, col2 = st.columns(2)
    with col1: st.markdown(f"""<div class="card"><div style="text-align: center;"><div style="font-size: 2.5rem; color: #667eea;">🎯</div><h3>Accuracy</h3><p style="font-size: 2rem; font-weight: bold;">{feature_info.get('accuracy', 0)*100:.1f}%</p></div></div>""", unsafe_allow_html=True)
    with col2: st.markdown(f"""<div class="card"><div style="text-align: center;"><div style="font-size: 2.5rem; color: #667eea;">🏷️</div><h3>Classes</h3><p style="font-size: 2rem; font-weight: bold;">{len(feature_info['target_classes'])}</p></div></div>""", unsafe_allow_html=True)

elif menu == "ℹ️ About":
    st.markdown("""<div class="card"><h3 style="color: #667eea;">🎯 Our Mission</h3><p>ThyroidCare AI is designed to assist healthcare professionals in early detection and diagnosis of thyroid conditions using advanced machine learning algorithms.</p></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="info-box"><strong>⚠️ Disclaimer:</strong><br>This system is intended as a decision support tool only. It should not replace professional medical advice, diagnosis, or treatment.</div>""", unsafe_allow_html=True)

# ===================================================================
# Footer
# ===================================================================
st.markdown("---")
st.markdown("""<div class="footer"><p>© 2026 ThyroidCare AI | Advanced Thyroid Disease Prediction System</p><p style="font-size: 0.9rem;">Developed with ❤️ using Streamlit & Machine Learning</p></div>""", unsafe_allow_html=True)