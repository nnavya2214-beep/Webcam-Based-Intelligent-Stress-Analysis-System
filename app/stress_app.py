"""
app/stress_app.py
StressVision AI — Facial Stress Analysis
Hybrid approach: CNN (emotion features) + pre-initialized LSTM (temporal stress)
"""
# ── Compatibility fix for pyparsing / httplib2 / TensorFlow ──────────────────
import pyparsing as _pp
if not hasattr(_pp, "DelimitedList"):
    _pp.DelimitedList = _pp.delimited_list

import os
import sys
import datetime
import numpy as np
import cv2
from PIL import Image
import streamlit as st

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.models import load_model

# Add project root so utils imports work
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from utils.stress_lstm import StressLSTMPredictor, EMOTIONS, STRESS_HIGH, STRESS_MEDIUM

try:
    import plotly.graph_objects as go
    import plotly.express as px
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StressVision AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
EMOTION_COLORS = {
    "Angry":    "#EF4444",
    "Disgust":  "#8B5CF6",
    "Fear":     "#6366F1",
    "Happy":    "#10B981",
    "Sad":      "#3B82F6",
    "Surprise": "#F59E0B",
    "Neutral":  "#6B7280",
}
EMOTION_EMOJIS = {
    "Angry": "😠", "Disgust": "🤢", "Fear": "😨",
    "Happy": "😊", "Sad": "😢", "Surprise": "😲", "Neutral": "😐",
}
LEVEL_COLORS = {"low": "#10B981", "medium": "#F59E0B", "high": "#EF4444"}
LEVEL_EMOJI  = {"low": "🟢", "medium": "🟡", "high": "🔴"}

RECOMMENDATIONS = {
    "Angry" : [
        "Count slowly to 10 before reacting.",
        "Splash cold water on your face.",
        "Write down what's frustrating you, then tear it up.",
    ],
    "Fear"  : [
        "Ground yourself: name 5 things you can see right now.",
        "Box breathing — 4s inhale, 4s hold, 4s exhale, 4s hold.",
        "Call or text someone you trust.",
    ],
    "Sad"   : [
        "Allow yourself to feel it — suppression makes it worse.",
        "Step outside for 10 minutes of sunlight.",
        "Put on music that usually lifts your mood.",
    ],
    "Disgust": [
        "Shift your focus to something pleasant or neutral.",
        "Take a brisk 5-minute walk.",
        "Practice a quick mindfulness scan of your body.",
    ],
    "Happy"  : ["Great state! Share your energy with someone.", "Keep doing what you're doing."],
    "Surprise": ["Take a breath and assess — is this good or bad news?"],
    "Neutral" : ["Good baseline. Stay hydrated and keep going.", "Consider a short stretch break."],
}
GENERAL_HIGH_TIPS = [
    "Take a 5-minute break — step away from the screen.",
    "Slow, deep breaths: breathe in for 4 counts, out for 6.",
    "Drink a glass of water now.",
    "Do a quick body scan: relax your jaw, shoulders, and hands.",
]

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Theme-friendly, compact, light visual polish
# ─────────────────────────────────────────────────────────────────────────────
def _load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .block-container {
        padding: 0.75rem 1.5rem 1rem !important;
        max-width: 1000px;
        margin: 0 auto;
    }

    .sv-header {
        display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;
        gap: 10px; padding-bottom: 0.75rem; margin-bottom: 0.75rem;
        border-bottom: 1px solid var(--border-color);
    }
    .sv-header-left { display: flex; align-items: baseline; gap: 10px; }
    .sv-title { font-size: 1.25rem; font-weight: 700; margin: 0; color: var(--text-color); letter-spacing: -0.02em; }
    .sv-subtitle { font-size: 0.8rem; color: var(--text-color); opacity: 0.7; margin: 0; }
    .sv-badge {
        font-size: 0.65rem; font-weight: 600; padding: 4px 10px; border-radius: 8px;
        background: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid var(--border-color);
    }

    .sv-section-title {
        font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.05em; color: var(--text-color); opacity: 0.6;
        margin: 0 0 0.5rem 0;
    }

    .sv-strip {
        display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap;
        padding: 0.65rem 1rem; border-radius: 10px; margin-bottom: 0.5rem;
        border-left: 4px solid;
        background: var(--secondary-background-color);
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .sv-strip-low   { border-left-color: #10B981; }
    .sv-strip-med   { border-left-color: #F59E0B; }
    .sv-strip-high  { border-left-color: #EF4444; }
    .sv-strip-label { font-size: 0.65rem; font-weight: 600; text-transform: uppercase; opacity: 0.7; color: var(--text-color); }
    .sv-strip-value { font-size: 1.2rem; font-weight: 700; color: var(--text-color); }
    .sv-strip-meta { font-size: 0.75rem; opacity: 0.65; margin-top: 2px; color: var(--text-color); }

    .sv-scores { display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 0.5rem 0; }
    .sv-score-box {
        min-width: 64px; padding: 0.5rem 0.85rem; border-radius: 10px;
        background: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .sv-score-box .n { font-size: 1.1rem; font-weight: 700; color: var(--text-color); }
    .sv-score-box .l { font-size: 0.6rem; font-weight: 600; text-transform: uppercase; opacity: 0.6; color: var(--text-color); }

    .sv-tips { margin: 0; padding-left: 1rem; color: var(--text-color); }
    .sv-tips li { font-size: 0.8rem; line-height: 1.45; margin-bottom: 0.3rem; opacity: 0.9; }

    .sv-sidebar-label { font-size: 0.65rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.6; margin-bottom: 0.35rem; color: var(--text-color); }

    .sv-main-card {
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 1.1rem;
        background: var(--background-color);
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .js-plotly-plot { max-height: 200px !important; }
    div[data-testid="stImage"] img {
        border-radius: 10px;
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING — best checkpoints first, then ensemble all available
# ─────────────────────────────────────────────────────────────────────────────
_MODEL_ORDER = (
    "best.h5",
    "emotion_model_best.h5",
    "emotion_model.h5",
    "emotion_model_fast.h5",
    "final.h5",
)


@st.cache_resource
def load_emotion_models():
    """Load all available emotion models for ensemble. Best checkpoints first."""
    roots = [_ROOT, os.path.join(_ROOT, "..")]
    loaded = []
    for base in roots:
        for name in _MODEL_ORDER:
            p = os.path.join(base, "models", name)
            if os.path.exists(p) and not any(n == name for _, n in loaded):
                try:
                    m = load_model(p)
                    loaded.append((m, name))
                except Exception:
                    continue
    return loaded


@st.cache_resource
def load_face_detector():
    cc_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cc_path)

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def detect_faces(bgr, detector):
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return list(faces) if len(faces) > 0 else []


def preprocess_face(face_bgr, use_clahe=True):
    """Resize face to 48x48, optional CLAHE for contrast."""
    try:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            gray = clahe.apply(gray)
        img = cv2.resize(gray, (48, 48)).astype("float32") / 255.0
        return img.reshape(1, 48, 48, 1)
    except Exception:
        return None


def predict_face_enhanced(face_bgr, models, n_crops=2, use_flip_tta=True):
    """
    Best prediction: ensemble over all models + multi-crop + optional horizontal flip TTA.
    models: list of (model, name) or single model.
    """
    if not isinstance(models, (list, tuple)):
        models = [(models, "")]
    model_list = [m[0] for m in models]
    h, w = face_bgr.shape[:2]
    all_probs = []

    for model in model_list:
        probs_list = []
        # Crop 1
        x = preprocess_face(face_bgr, use_clahe=True)
        if x is not None:
            probs_list.append(model.predict(x, verbose=0)[0])
        # Crop 2: slight zoom-out
        if n_crops >= 2 and h > 20 and w > 20:
            pad_h, pad_w = int(h * 0.08), int(w * 0.08)
            padded = cv2.copyMakeBorder(face_bgr, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
            cy, cx = padded.shape[0] // 2, padded.shape[1] // 2
            half = min(cy, cx, h // 2 + pad_h, w // 2 + pad_w)
            crop = padded[cy - half : cy + half, cx - half : cx + half]
            if crop.size > 0:
                x2 = preprocess_face(crop, use_clahe=True)
                if x2 is not None:
                    probs_list.append(model.predict(x2, verbose=0)[0])
        # Flip TTA
        if use_flip_tta and face_bgr.size > 0:
            flipped = cv2.flip(face_bgr, 1)
            xf = preprocess_face(flipped, use_clahe=True)
            if xf is not None:
                probs_list.append(model.predict(xf, verbose=0)[0])
        if probs_list:
            all_probs.append(np.mean(probs_list, axis=0))

    if not all_probs:
        return None
    probs = np.mean(all_probs, axis=0).astype(np.float32)
    probs = probs / (probs.sum() + 1e-8)
    return probs

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def _gauge(score, level):
    color = LEVEL_COLORS[level]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "", "font": {"size": 22, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickfont": {"size": 9}},
            "bar": {"color": color, "thickness": 0.18},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(16,185,129,.15)"},
                {"range": [40, 65], "color": "rgba(245,158,11,.15)"},
                {"range": [65, 100], "color": "rgba(239,68,68,.15)"},
            ],
            "threshold": {"line": {"color": color, "width": 2}, "thickness": 0.7, "value": score},
        },
    ))
    fig.update_layout(height=160, margin=dict(l=5, r=5, t=5, b=5),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(size=10))
    return fig


def _emotion_bar(per_emotion):
    labels = list(per_emotion.keys())
    values = [v * 100 for v in per_emotion.values()]
    colors = [EMOTION_COLORS[e] for e in labels]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v:.0f}" for v in values], textposition="outside", textfont=dict(size=9),
    ))
    fig.update_layout(
        yaxis=dict(range=[0, 105], tickfont=dict(size=9)),
        height=180, margin=dict(l=5, r=5, t=4, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, font=dict(size=9),
    )
    return fig


def _trend_chart(history):
    times, scores = [h["time"] for h in history], [h["score"] for h in history]
    colors = [LEVEL_COLORS[h["level"]] for h in history]
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=40, fillcolor="rgba(16,185,129,.08)", line_width=0)
    fig.add_hrect(y0=40, y1=65, fillcolor="rgba(245,158,11,.08)", line_width=0)
    fig.add_hrect(y0=65, y1=100, fillcolor="rgba(239,68,68,.08)", line_width=0)
    fig.add_trace(go.Scatter(
        x=times, y=scores, mode="lines+markers",
        line=dict(color="#6366F1", width=2),
        marker=dict(size=6, color=colors, line=dict(width=1, color="white")),
    ))
    fig.update_layout(
        yaxis=dict(title="Score", range=[0, 100], tickfont=dict(size=9)),
        xaxis=dict(title="Reading", tickfont=dict(size=9)),
        height=220, margin=dict(l=8, r=8, t=4, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(size=9),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS (structured)
# ─────────────────────────────────────────────────────────────────────────────
def _stress_strip(score, level, dominant):
    """Compact stress result strip."""
    emoji = LEVEL_EMOJI[level]
    msg = {"high": "Elevated stress — take a moment.", "medium": "Moderate stress — a short break may help.", "low": "Calm and composed."}[level]
    st.markdown(f"""
    <div class="sv-strip sv-strip-{level}">
        <div>
            <div class="sv-strip-label">Stress level</div>
            <div class="sv-strip-value" style="color:{LEVEL_COLORS[level]};">{emoji} {level.upper()}</div>
            <div class="sv-strip-meta">{EMOTION_EMOJIS.get(dominant,'')} {dominant} · {score:.0f}/100</div>
        </div>
        <div style="flex:1; font-size:0.85rem; opacity:0.85; color: var(--text-color);">{msg}</div>
    </div>
    """, unsafe_allow_html=True)


def _score_row(score, breakdown):
    lstm_s = breakdown.get("lstm_score")
    rule_s = breakdown.get("rule_score", score)
    wl = breakdown.get("window_len", 0)
    color = LEVEL_COLORS["high"] if score >= 65 else LEVEL_COLORS["medium"] if score >= 40 else LEVEL_COLORS["low"]
    row = f'<div class="sv-scores"><div class="sv-score-box"><div class="n" style="color:{color};">{score:.0f}</div><div class="l">Hybrid</div></div><div class="sv-score-box"><div class="n">{rule_s:.0f}</div><div class="l">Rule</div></div>'
    if lstm_s is not None:
        row += f'<div class="sv-score-box"><div class="n">{lstm_s:.0f}</div><div class="l">LSTM ({wl})</div></div>'
    row += "</div>"
    st.markdown(row, unsafe_allow_html=True)


def _tips_block(level, dominant):
    tips = (GENERAL_HIGH_TIPS if level == "high" else []) + RECOMMENDATIONS.get(dominant, [])
    items = "".join(f"<li>{t}</li>" for t in tips[:4])
    st.markdown(f'<p class="sv-section-title">Recommendations</p><ul class="sv-tips">{items}</ul>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CORE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def analyse(img_bgr, models, detector, predictor, temporal=False):
    """
    Detect faces, run enhanced CNN (multi-crop + CLAHE) + LSTM, return result dicts per face.
    Applies confidence dampening: low emotion confidence → stress score pulled toward 50.
    """
    results = []
    faces   = detect_faces(img_bgr, detector)
    for (x, y, w, h) in faces:
        face_roi = img_bgr[y:y+h, x:x+w]
        probs    = predict_face_enhanced(face_roi, models, n_crops=2, use_flip_tta=True)
        if probs is None:
            continue
        label_i  = int(np.argmax(probs))
        dominant = EMOTIONS[label_i]
        conf     = float(probs[label_i])

        if temporal:
            predictor.update(probs)
            score, level, breakdown = predictor.predict()
        else:
            score, level, breakdown = predictor.instant_stress(probs)

        # Confidence dampening: if emotion prediction is uncertain, don't over-call stress
        if conf < 0.45:
            score = score * (conf / 0.45) + 50.0 * (1.0 - conf / 0.45)
            score = float(np.clip(score, 0.0, 100.0))
            level = "high" if score >= 65 else "medium" if score >= 40 else "low"
            breakdown["rule_score"] = round(score, 1)

        breakdown["dominant"] = dominant
        breakdown["confidence"] = round(conf * 100, 1)
        results.append({
            "bbox"     : (x, y, w, h),
            "probs"    : probs,
            "dominant" : dominant,
            "conf"     : conf * 100,
            "score"    : score,
            "level"    : level,
            "breakdown": breakdown,
        })
    return results, faces

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP — Structured layout
# ─────────────────────────────────────────────────────────────────────────────
def main():
    _load_css()

    if "predictor" not in st.session_state:
        st.session_state.predictor = StressLSTMPredictor()
    if "history" not in st.session_state:
        st.session_state.history = []
    predictor = st.session_state.predictor

    # ── Sidebar (compact) ────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<p class="sv-sidebar-label">System</p>', unsafe_allow_html=True)
        st.caption("Face → CNN (7 emotions) → LSTM + rules → stress 0–100")
        st.markdown("---")
        st.markdown('<p class="sv-sidebar-label">Model</p>', unsafe_allow_html=True)
        models_loaded = load_emotion_models()
        if models_loaded:
            model_display = f"Ensemble ({len(models_loaded)})" if len(models_loaded) > 1 else models_loaded[0][1]
            st.caption(model_display)
        else:
            st.error("Missing. Run: `python train_simple.py`")
        st.markdown('<p class="sv-sidebar-label">Session</p>', unsafe_allow_html=True)
        n = len(st.session_state.history)
        st.caption(f"Readings: {n}" + (f" · Avg: {np.mean([h['score'] for h in st.session_state.history]):.0f}" if n else ""))
        if st.button("Reset session"):
            st.session_state.history = []
            st.session_state.predictor.reset()
            st.rerun()
        st.markdown('<p class="sv-sidebar-label">Thresholds</p>', unsafe_allow_html=True)
        st.caption(f"Low &lt;{STRESS_MEDIUM:.0f}  Med &lt;{STRESS_HIGH:.0f}  High ≥{STRESS_HIGH:.0f}")

    if not models_loaded:
        st.stop()
    detector = load_face_detector()

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sv-header">
        <div class="sv-header-left">
            <h1 class="sv-title">StressVision AI</h1>
            <span class="sv-badge">CNN + LSTM</span>
        </div>
        <p class="sv-subtitle">Facial stress analysis · Hybrid temporal model</p>
    </div>
    """, unsafe_allow_html=True)

    tab_analyzer, tab_trend = st.tabs(["Analyzer", "Session trend"])

    # ══ TAB: Analyzer — single card, no scroll: input → image + output in one row ══
    with tab_analyzer:
        st.markdown('<div class="sv-main-card">', unsafe_allow_html=True)
        # Row 1: Input (mode + track)
        r1c1, r1c2 = st.columns([3, 1])
        with r1c1:
            mode = st.radio("", ["Camera", "Upload"], horizontal=True, label_visibility="collapsed")
        with r1c2:
            temporal_mode = st.checkbox("Track session", value=True, label_visibility="collapsed", help="LSTM over time")
        # Row 2: Camera or file (compact)
        uploaded = st.camera_input("") if mode == "Camera" else st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            with st.spinner("Analyzing…"):
                results, faces = analyse(img_bgr, models_loaded, detector, predictor, temporal=temporal_mode)

            if not results:
                st.image(image, width=240)
                st.warning("No face detected. Use a clear, frontal face.")
            else:
                vis = img_bgr.copy()
                for r in results:
                    x, y, w, h = r["bbox"]
                    c = LEVEL_COLORS[r["level"]]
                    bgr = tuple(int(c.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))
                    cv2.rectangle(vis, (x, y), (x+w, y+h), bgr, 2)
                img_display = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                r0 = results[0]

                # Single row: Image | Stress + scores | Gauge | Emotions (no scroll)
                st.markdown('<div class="sv-main-card">', unsafe_allow_html=True)
                col_img, col_stress, col_gauge, col_emo = st.columns([1, 1, 1, 1])
                with col_img:
                    st.image(img_display, width=200)
                    st.caption(f"{len(results)} face(s)")
                with col_stress:
                    _stress_strip(r0["score"], r0["level"], r0["dominant"])
                    if r0["breakdown"].get("confidence") is not None:
                        st.caption(f"Conf: {r0['breakdown']['confidence']:.0f}%")
                    _score_row(r0["score"], r0["breakdown"])
                with col_gauge:
                    if _PLOTLY:
                        st.plotly_chart(_gauge(r0["score"], r0["level"]), use_container_width=True)
                    else:
                        st.metric("Stress", f"{r0['score']:.0f}/100")
                with col_emo:
                    if _PLOTLY:
                        st.plotly_chart(_emotion_bar(r0["breakdown"]["per_emotion"]), use_container_width=True)
                    else:
                        for e, p in r0["breakdown"]["per_emotion"].items():
                            st.caption(f"{EMOTION_EMOJIS[e]} {e}: {p*100:.0f}%")
                st.markdown('</div>', unsafe_allow_html=True)

                # Tips in expander so layout stays compact
                with st.expander("Tips & recommendations"):
                    _tips_block(r0["level"], r0["dominant"])
                if r0["level"] == "high":
                    st.error("High stress — take a break. Breathe: IN 4s, HOLD 4s, OUT 6s.")

                if temporal_mode:
                    for r in results:
                        st.session_state.history.append({
                            "time": len(st.session_state.history) + 1,
                            "score": r["score"], "level": r["level"], "dominant": r["dominant"],
                            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                        })

    # ══ TAB: Session trend (compact) ═══════════════════════════════════════════
    with tab_trend:
        h = st.session_state.history
        if not h:
            st.info("Turn on **Track session** and analyze an image to see the trend.")
        else:
            scores = [x["score"] for x in h]
            st.markdown('<div class="sv-main-card">', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Readings", len(h))
            m2.metric("Current", f"{scores[-1]:.0f}")
            m3.metric("Avg", f"{np.mean(scores):.0f}")
            m4.metric("Peak", f"{max(scores):.0f}")
            if _PLOTLY:
                st.plotly_chart(_trend_chart(h), use_container_width=True)
            else:
                st.line_chart({r["time"]: r["score"] for r in h})
            st.markdown('</div>', unsafe_allow_html=True)
            for rec in reversed(h[-10:]):
                st.caption(f"#{rec['time']} {rec['timestamp']} — {LEVEL_EMOJI[rec['level']]} {rec['score']:.0f} · {EMOTION_EMOJIS[rec['dominant']]} {rec['dominant']}")
            if len(scores) >= 3:
                d = scores[-1] - scores[0]
                if d > 10:
                    st.warning(f"Stress +{d:.0f} this session.")
                elif d < -10:
                    st.success(f"Stress {d:.0f} this session.")
            if scores[-1] >= STRESS_HIGH:
                st.error("High stress — take a short break.")


if __name__ == "__main__":
    main()
