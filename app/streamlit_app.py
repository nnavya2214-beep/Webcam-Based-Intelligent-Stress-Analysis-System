"""
app/streamlit_app.py
PREMIUM VERSION - Ultra-Modern Professional UI
"""
# Fix pyparsing 3.x compatibility: httplib2 (TensorFlow dep) expects DelimitedList
import pyparsing as _pp
if not hasattr(_pp, "DelimitedList"):
    _pp.DelimitedList = _pp.delimited_list

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import tensorflow as tf
import google.generativeai as genai
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="EmotiVision AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CONSTANTS
# ==========================================

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
EMOTION_EMOJIS = ['😠', '🤢', '😨', '😊', '😢', '😲', '😐']
EMOTION_COLORS = ['#FF6B6B', '#9B59B6', '#34495E', '#F1C40F', '#3498DB', '#1ABC9C', '#95A5A6']

# Stress: CNN + rule-based algorithm (see utils/stress.py)
try:
    from utils.stress import stress_score_from_cnn, rule_based_stress_level
except ImportError:
    def stress_score_from_cnn(probs):
        w = np.array([1.0, 0.9, 1.0, -0.8, 0.95, -0.2, 0.1])
        p = np.asarray(probs, dtype=np.float64).flatten()
        if len(p) != 7:
            return 50.0
        raw = np.clip(np.dot(w, p), -1.0, 1.0)
        return float(np.clip(50.0 + 50.0 * raw, 0.0, 100.0))
    def rule_based_stress_level(score):
        return "high" if score >= 65 else "medium" if score >= 40 else "low"

# ==========================================
# (MOTIVATION_DATA dictionary removed - using LLM)
# ==========================================

# ==========================================
# PREMIUM CSS STYLING
# ==========================================

def load_premium_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap');
        
        /* Global Reset */
        * {
            box-sizing: border-box;
        }
        
        /* Typography */
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
            -webkit-font-smoothing: antialiased;
        }
        
        /* Layout & Spacing Defaults */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 4rem !important;
        }
        
        /* Hero Header - High-end Glassmorphism */
        .hero-container {
            text-align: center;
            padding: 60px 20px;
            background: var(--background-color);
            border-radius: 24px;
            margin-bottom: 40px;
            border: 1px solid var(--secondary-background-color);
            box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.08);
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        
        .hero-title {
            font-family: 'Outfit', sans-serif;
            font-size: 3.5rem;
            font-weight: 800;
            color: var(--text-color);
            margin-bottom: 12px;
            letter-spacing: -1.5px;
            line-height: 1.1;
        }
        
        .hero-title-gradient {
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .hero-subtitle {
            font-size: 1.15rem;
            color: var(--text-color);
            opacity: 0.7;
            font-weight: 400;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            margin-bottom: 24px;
        }
        
        .hero-badge {
            display: inline-block;
            padding: 8px 20px;
            background: rgba(79, 70, 229, 0.08);
            border: 1px solid rgba(79, 70, 229, 0.2);
            border-radius: 100px;
            color: var(--text-color);
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        
        /* Premium Card System */
        .premium-card {
            background: var(--background-color);
            border-radius: 24px;
            padding: 32px;
            border: 1px solid var(--secondary-background-color);
            box-shadow: 0 4px 20px -5px rgba(0, 0, 0, 0.05);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .premium-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 30px -10px rgba(79, 70, 229, 0.15);
            border-color: rgba(79, 70, 229, 0.3);
        }
        
        /* Feature Cards */
        .feature-card {
            background: var(--background-color);
            border-radius: 20px;
            padding: 28px;
            border: 1px solid var(--secondary-background-color);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 15px -5px rgba(0, 0, 0, 0.05);
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        
        .feature-card:hover {
            border-color: rgba(79, 70, 229, 0.4);
            transform: translateY(-4px);
            box-shadow: 0 12px 25px -10px rgba(0, 0, 0, 0.1);
        }
        
        .feature-icon {
            font-size: 2.8rem;
            margin-bottom: 16px;
        }
        
        .feature-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--text-color);
            margin-bottom: 10px;
            letter-spacing: -0.5px;
        }
        
        .feature-text {
            color: var(--text-color);
            opacity: 0.7;
            line-height: 1.6;
            font-size: 0.95rem;
            font-weight: 400;
        }
        
        /* Section Headers */
        .section-header {
            font-family: 'Outfit', sans-serif;
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-color);
            margin: 40px 0 24px 0;
            padding-left: 16px;
            border-left: 4px solid #4F46E5;
            letter-spacing: -0.5px;
            line-height: 1.2;
        }
        
        /* Emotion List Card */
        .emotion-list-card {
            background: var(--background-color);
            border-radius: 20px;
            padding: 24px;
            border: 1px solid var(--secondary-background-color);
            box-shadow: 0 4px 15px -5px rgba(0, 0, 0, 0.05);
            text-align: center;
            transition: all 0.2s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        
        .emotion-list-card:hover {
            border-color: var(--text-color);
            opacity: 0.9;
            transform: translateY(-2px);
        }
        
        .emotion-list-icon {
            font-size: 3rem;
            margin-bottom: 12px;
        }
        
        .emotion-list-title {
            font-family: 'Inter', sans-serif;
            font-size: 1.1rem;
            font-weight: 600;
            letter-spacing: -0.2px;
        }
        
        /* Upload / Input Containers */
        .input-container {
            background: rgba(79, 70, 229, 0.03); 
            border: 2px dashed rgba(79, 70, 229, 0.3); 
            border-radius: 24px; 
            padding: 32px; 
            margin-bottom: 24px;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .input-container:hover {
            border-color: rgba(79, 70, 229, 0.6);
            background: rgba(79, 70, 229, 0.05); 
        }
        
        /* File Uploader - Custom Style Override */
        [data-testid="stFileUploader"] {
            background: var(--background-color);
            border: 1px solid var(--secondary-background-color);
            border-radius: 16px;
            padding: 24px;
            transition: all 0.2s ease;
            box-shadow: 0 2px 10px rgba(0,0,0,0.02);
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: #4F46E5;
            box-shadow: 0 4px 15px rgba(79, 70, 229, 0.1);
        }
        
        /* Streamlit Buttons - Primary Action Style */
        .stButton>button {
            background: #4F46E5;
            color: white;
            border: 1px solid #4F46E5;
            padding: 10px 24px;
            border-radius: 12px;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            transition: all 0.2s ease;
            box-shadow: 0 2px 8px rgba(79, 70, 229, 0.25);
        }
        
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
            color: white;
            border: 1px solid #4338CA;
            background: #4338CA;
        }
        
        /* Emotion Result Badge */
        .emotion-badge {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 16px 24px;
            background: var(--background-color);
            border: 1px solid var(--secondary-background-color);
            border-radius: 20px;
            font-family: 'Outfit', sans-serif;
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--text-color);
            margin: 16px 0;
            box-shadow: 0 8px 25px -8px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        .stress-alert-high {
            background: linear-gradient(135deg, rgba(220, 38, 38, 0.12) 0%, rgba(185, 28, 28, 0.08) 100%);
            border: 2px solid #DC2626;
            border-radius: 16px;
            padding: 20px 24px;
            margin: 20px 0;
            font-family: 'Inter', sans-serif;
            box-shadow: 0 4px 20px rgba(220, 38, 38, 0.2);
        }
        .stress-alert-high .stress-alert-title { color: #B91C1C; font-weight: 800; font-size: 1.25rem; margin-bottom: 8px; }
        .stress-alert-high .stress-alert-msg { color: #7F1D1D; line-height: 1.5; margin-bottom: 12px; }
        .stress-alert-high .stress-alert-tip { color: #991B1B; font-weight: 600; }
        .stress-indicator { font-size: 0.95rem; opacity: 0.85; margin-top: 8px; }
        
        /* Motivation Panel - Clean High-End Design */
        .motivation-panel {
            background: var(--background-color);
            border-radius: 20px;
            padding: 32px;
            margin-top: 24px;
            margin-bottom: 24px;
            border: 1px solid var(--secondary-background-color);
            border-top: 4px solid;
            box-shadow: 0 10px 30px -10px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
        }
        
        .motivation-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 20px;
        }
        
        .motivation-icon {
            font-size: 1.8rem;
            line-height: 1;
        }
        
        .motivation-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--text-color);
            opacity: 0.6;
        }
        
        .motivation-quote {
            font-family: 'Inter', sans-serif;
            font-size: 1.15rem;
            font-style: italic;
            font-weight: 500;
            color: var(--text-color);
            line-height: 1.6;
            margin-bottom: 24px;
            padding-left: 16px;
            border-left: 3px solid rgba(79, 70, 229, 0.4);
        }
        
        .motivation-divider {
            height: 1px;
            background: var(--secondary-background-color);
            margin: 20px 0;
            opacity: 0.5;
        }
        
        .motivation-tip, .motivation-activity {
            font-family: 'Inter', sans-serif;
            font-size: 1rem;
            color: var(--text-color);
            opacity: 0.85;
            margin-bottom: 12px;
            line-height: 1.6;
        }
        
        .motivation-tip strong, .motivation-activity strong {
            color: var(--text-color);
            opacity: 1;
            font-weight: 700;
        }
        
        /* Success Message Box */
        .success-box {
            background: rgba(16, 185, 129, 0.05); 
            border: 1px solid rgba(16, 185, 129, 0.2); 
            border-radius: 16px; 
            padding: 16px 24px; 
            margin-bottom: 24px;
            text-align: center;
            color: #10B981;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        
        /* Streamlit Overrides */
        .stAlert {
            border-radius: 12px;
        }
        
        .stImage > img {
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            border: 1px solid var(--secondary-background-color);
        }
        
        hr {
            border: none;
            height: 1px;
            background: var(--secondary-background-color);
            margin: 32px 0;
            opacity: 0.5;
        }
        
        /* Sidebar Navigation Spacing */
        [data-testid="stSidebar"] [role="radiogroup"] {
            gap: 8px;
        }
        
        [data-testid="stSidebar"] [role="radiogroup"] label {
            padding: 12px 16px;
            border-radius: 12px;
            transition: all 0.2s ease;
        }
        
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(79, 70, 229, 0.08);
        }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# HELPER CLASSES
# ==========================================

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect_faces(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

@st.cache_resource
def load_emotion_model():
    possible_paths = [
        'models/best.h5',
        '../models/best.h5',
        'models/emotion_model.h5',
        '../models/emotion_model.h5'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                model = load_model(path)
                return model, "Our Best Models (83% Accuracy)"
            except:
                continue
    return None, None
    possible_paths = [
        'models/best.h5',
        '../models/best.h5',
        'models/emotion_model.h5',
        '../models/emotion_model.h5'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                model = load_model(path)
                return model, path
            except:
                continue
    return None, None

@st.cache_data(show_spinner=False, ttl=3600)
def generate_motivation(emotion_name, api_key):
    """Generate dynamic motivation using Gemini API, cached to avoid redundant calls."""
    try:
        genai.configure(api_key=api_key)
        
        # Determine appropriate model
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
        except:
            model = genai.GenerativeModel('gemini-1.5-flash')

        prompt = f"""
        You are an empathetic, insightful AI wellness coach. 
        The user's facial expression currently indicates they are feeling: {emotion_name}.
        
        Please provide a short, uplifting message tailored to this emotion. 
        Your response MUST be a valid JSON object with the following exact keys:
        - "quote": A highly relevant short quote (include the author).
        - "tip": A 1-sentence practical wellness tip.
        - "activity": A 1-sentence suggested daily activity.
        
        Do not include any markdown formatting like ```json or newlines outside the JSON format. Just valid JSON.
        """
        
        response = model.generate_content(prompt)
        text_response = response.text.strip()
        
        # Clean up possible markdown code blocks often returned by LLMs
        if text_response.startswith('```json'):
            text_response = text_response[7:]
        if text_response.startswith('```'):
            text_response = text_response[3:]
        if text_response.endswith('```'):
            text_response = text_response[:-3]
            
        return json.loads(text_response.strip())
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return None

def show_motivation_panel(emotion_name, api_key):
    """Render a rich motivation panel for the detected emotion generated via LLM."""
    
    # Generic stylistic fallbacks based on emotion since we removed the dictionary
    style_map = {
        'Angry': ('#FF6B6B', 'rgba(255,107,107,0.4)', '🌊'),
        'Disgust': ('#9B59B6', 'rgba(155,89,182,0.4)', '🌸'),
        'Fear': ('#4A90A4', 'rgba(74,144,164,0.4)', '🦁'),
        'Happy': ('#F1C40F', 'rgba(241,196,15,0.4)', '✨'),
        'Sad': ('#3498DB', 'rgba(52,152,219,0.4)', '🌈'),
        'Surprise': ('#1ABC9C', 'rgba(26,188,156,0.4)', '🎉'),
        'Neutral': ('#95A5A6', 'rgba(149,165,166,0.4)', '🧘')
    }
    
    border, glow, icon = style_map.get(emotion_name, ('#667eea', 'rgba(102,126,234,0.4)', '💡'))
    
    if not api_key:
        st.markdown(f"""
            <div class="motivation-panel" style="border-left-color: {border};">
                <div class="motivation-header">
                    <span class="motivation-icon">{icon}</span>
                    <span class="motivation-title">💡 Motivation · API Key Required</span>
                </div>
                <div class="motivation-tip" style="color: #ff6b6b !important;">
                    Please enter your Google Gemini API Key in the sidebar to generate dynamic wellness insights for your emotion.
                </div>
            </div>
        """, unsafe_allow_html=True)
        return

    with st.spinner("Generating personalized insights..."):
        llm_data = generate_motivation(emotion_name, api_key)
        
    if llm_data:
        quote = llm_data.get('quote', "Keep going, you're doing great.")
        tip = llm_data.get('tip', "Take a deep breath and pause for a moment.")
        activity = llm_data.get('activity', "Take a short walk to reset your mind.")
        
        st.markdown(f"""
            <div class="motivation-panel" style="border-left-color: {border};">
                <div class="motivation-header">
                    <span class="motivation-icon">{icon}</span>
                    <span class="motivation-title">💡 Coach · Feeling {emotion_name}</span>
                </div>
                <div class="motivation-quote">"{quote}"</div>
                <div class="motivation-divider"></div>
                <div class="motivation-tip"><strong>Wellness Tip:</strong> {tip}</div>
                <div class="motivation-activity"><strong>Try This:</strong> {activity}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Failed to generate motivation. Please check your API key.")

def preprocess_face(face_img):
    try:
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        img = cv2.resize(face_img, (48, 48))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
        return img
    except Exception as e:
        st.error(f"Error processing face: {e}")
        return None

# ==========================================
# MAIN APPLICATION
# ==========================================

def main():
    load_premium_css()
    
    # Hero Header
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">
                🎭 <span class="hero-title-gradient">EmotiVision AI</span>
            </div>
            <p class="hero-subtitle">Next-Gen Emotion Recognition System</p>
            <span class="hero-badge">Powered by Deep Learning</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Load resources
    model, model_path = load_emotion_model()
    detector = FaceDetector()
    
    # Sidebar
    with st.sidebar:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 80px;">🧠</div>
            </div>
        """, unsafe_allow_html=True)
        
        default_api_key = os.getenv("GEMINI_API_KEY", "AIzaSyCMRx_dBk6pX4rZagC51Vd3bIB7_DNWHOc")
        gemini_api_key = st.text_input("Gemini API Key", value=default_api_key, type="password", help="Enter your Google Gemini API Key for dynamic motivation")
        
        st.markdown("### 🎯 Navigation")
        page = st.radio(
            "",
            ["🏠 Home", "📸 Analyzer"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.caption("🎓 Final Year Project 2024")
    
    # ==========================================
    # PAGE: HOME
    # ==========================================
    if page == "🏠 Home":
        # Features Section
        st.markdown('<h2 class="section-header">🚀 How it Works</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class="feature-card">
                    <div class="feature-icon">📸</div>
                    <div class="feature-title">1. Upload</div>
                    <p class="feature-text">Simply upload a photo with a clear view of a face.</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="feature-card">
                    <div class="feature-icon">🧠</div>
                    <div class="feature-title">2. Detect</div>
                    <p class="feature-text">Our AI instantly analyzes the facial expression to detect emotions.</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="feature-card">
                    <div class="feature-icon">💡</div>
                    <div class="feature-title">3. Guide</div>
                    <p class="feature-text">Receive a personalized, uplifting wellness tip based on your mood.</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Emotion Classes
        st.markdown('<h2 class="section-header">😊 Emotion Classes</h2>', unsafe_allow_html=True)
        
        cols = st.columns(7)
        for idx, (emoji, emotion, color) in enumerate(zip(EMOTION_EMOJIS, EMOTIONS, EMOTION_COLORS)):
            with cols[idx]:
                st.markdown(f"""
                    <div class="emotion-list-card">
                        <div class="emotion-list-icon">{emoji}</div>
                        <div class="emotion-list-title" style="color: {color};">{emotion}</div>
                    </div>
                """, unsafe_allow_html=True)
    
    # ==========================================
    # PAGE: ANALYZER
    # ==========================================
    elif page == "📸 Analyzer":
        st.markdown('<h2 class="section-header">📸 Capture Your Expression</h2>', unsafe_allow_html=True)
        
        st.markdown("""
            <div class="input-container">
                <p style="font-size: 1.15rem; color: var(--text-color); opacity: 0.8; margin: 0; line-height: 1.5;">
                    Choose your input method below to instantly detect your emotion and receive personalized wellness tips powered by AI.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        input_method = st.radio(
            "Select Input Method:",
            ["📸 Take Photo", "📁 Upload Image"],
            horizontal=True
        )
        
        uploaded_file = None
        
        if input_method == "📸 Take Photo":
            uploaded_file = st.camera_input("Take a picture of your expression!")
        else:
            uploaded_file = st.file_uploader(
                "Choose an image",
                type=['jpg', 'jpeg', 'png'],
                help="Supported: JPG, JPEG, PNG | Max size: 200MB",
                label_visibility="collapsed"
            )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            img_array = np.array(image.convert('RGB'))
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                st.markdown("### 📷 Original Image")
                st.image(image, use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            faces = detector.detect_faces(img_bgr)
            
            with col2:
                st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                st.markdown("### 🎯 Detection Results")
                
                if len(faces) == 0:
                    st.warning("⚠️ No faces detected. Please upload a clearer image with visible faces.")
                else:
                    st.markdown(f"""
                        <div class="success-box">
                            <span style="font-size: 1.2rem;">✨</span>
                            <span>Successfully detected {len(faces)} face(s)</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    for i, (x, y, w, h) in enumerate(faces):
                        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (79, 70, 229), 4)
                        
                        face_roi = img_bgr[y:y+h, x:x+w]
                        processed_face = preprocess_face(face_roi)
                        
                        if processed_face is not None:
                            preds = model.predict(processed_face, verbose=0)[0]
                            label_idx = np.argmax(preds)
                            conf = preds[label_idx] * 100
                            
                            st.markdown(f"""
                                <div class="emotion-badge" style="border-top: 4px solid {EMOTION_COLORS[label_idx]};">
                                    <span style="font-size: 2rem;">{EMOTION_EMOJIS[label_idx]}</span>
                                    <span style="flex-grow: 1;">Face #{i+1}: <span style="color: {EMOTION_COLORS[label_idx]};">{EMOTIONS[label_idx]}</span></span>
                                    <span style="opacity: 0.6; font-size: 1.1rem; font-weight: 500;">{conf:.1f}% Match</span>
                                </div>
                            """, unsafe_allow_html=True)

                            # Stress: CNN probabilities + rule-based algorithm
                            stress_score = stress_score_from_cnn(preds)
                            stress_level = rule_based_stress_level(stress_score)
                            st.markdown(f"""
                                <div class="stress-indicator">
                                    📊 <strong>Stress estimate</strong> (CNN + rule-based): <strong>{stress_score:.0f}/100</strong> — <span style="color: {'#DC2626' if stress_level == 'high' else '#F59E0B' if stress_level == 'medium' else '#059669'};">{stress_level.upper()}</span>
                                </div>
                            """, unsafe_allow_html=True)

                            if stress_level == "high":
                                st.markdown("""
                                    <div class="stress-alert-high">
                                        <div class="stress-alert-title">⚠️ High stress detected</div>
                                        <div class="stress-alert-msg">Our system (CNN + rule-based check) suggests elevated stress. Please take a short break, breathe slowly, or do a quick activity you enjoy.</div>
                                        <div class="stress-alert-tip">💡 Tip: If this persists, consider talking to someone or doing a 5-minute walk.</div>
                                    </div>
                                """, unsafe_allow_html=True)
                                st.error("**High stress alert** — Consider taking a break or a short walk.")

                            show_motivation_panel(EMOTIONS[label_idx], gemini_api_key)
                            
                            st.progress(conf / 100)
                
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()