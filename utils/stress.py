"""
utils/stress.py
Stress estimation: CNN emotion probabilities + rule-based algorithm.
Used by Streamlit app and realtime emotion script for high-stress alerts.
"""
import numpy as np

# Order: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
STRESS_WEIGHTS = np.array([1.0, 0.9, 1.0, -0.8, 0.95, -0.2, 0.1])
STRESS_HIGH_THRESHOLD = 65
STRESS_MEDIUM_THRESHOLD = 40


def stress_score_from_cnn(probs):
    """Compute stress score (0–100) from CNN emotion probabilities."""
    probs = np.asarray(probs, dtype=np.float64).flatten()
    if len(probs) != 7:
        return 50.0
    raw = np.dot(STRESS_WEIGHTS, probs)
    score = 50.0 + 50.0 * np.clip(raw, -1.0, 1.0)
    return float(np.clip(score, 0.0, 100.0))


def rule_based_stress_level(score):
    """Rule-based classifier: 'low' | 'medium' | 'high'."""
    if score >= STRESS_HIGH_THRESHOLD:
        return "high"
    if score >= STRESS_MEDIUM_THRESHOLD:
        return "medium"
    return "low"
