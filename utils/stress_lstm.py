"""
utils/stress_lstm.py
Pre-initialized LSTM for temporal stress prediction.

Architecture: CNN emotion probs (7-dim rolling window) → LSTM(16) → stress score
Pre-initialized weights encode the valence-arousal psychological model:
  Angry / Fear / Disgust / Sad  → raise stress cell state
  Happy / Neutral               → lower stress cell state
  Temporal memory               → sustained negative emotions accumulate

Combined (hybrid) score = 60% LSTM + 40% rule-based weighted sum.
"""

import numpy as np

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
WINDOW_SIZE = 15
HIDDEN_SIZE = 16

# Stress contribution weights per emotion (valence-arousal psychology)
# High arousal-negative  → high positive weight  (Angry, Fear)
# Low arousal-negative   → moderate weight        (Sad, Disgust)
# Positive valence       → negative weight        (Happy)
# Ambiguous / neutral    → small weight           (Surprise, Neutral)
STRESS_WEIGHTS = np.array([1.0, 0.7, 1.0, -0.9, 0.8, 0.15, -0.3], dtype=np.float32)

STRESS_HIGH   = 65.0
STRESS_MEDIUM = 40.0


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _tanh(x):
    return np.tanh(np.clip(x, -20.0, 20.0))


class StressLSTMPredictor:
    """
    Hybrid stress predictor: pre-initialized LSTM + rule-based ensemble.

    Usage
    -----
    predictor = StressLSTMPredictor()

    # For each new CNN emotion output (7 probabilities):
    predictor.update(probs)
    score, level, breakdown = predictor.predict()

    # For a single static image (no temporal context):
    score, level, breakdown = predictor.instant_stress(probs)
    """

    def __init__(self, window_size=WINDOW_SIZE):
        self.window_size = window_size
        self.input_size  = 7
        self.hidden_size = HIDDEN_SIZE
        self.buffer      = []
        self._init_weights()
        self.reset()

    # ------------------------------------------------------------------
    # Weight initialisation (pre-trained with domain knowledge)
    # ------------------------------------------------------------------
    def _init_weights(self):
        rng   = np.random.default_rng(2024)
        n     = self.input_size + self.hidden_size   # 23
        h     = self.hidden_size                      # 16
        noise = 0.06

        # Forget gate  ← bias high (remember stress history)
        self.W_f = rng.standard_normal((n, h)).astype(np.float32) * noise
        self.b_f = np.ones(h, dtype=np.float32) * 1.0

        # Input gate  ← opens when stress-related emotions appear
        self.W_i = rng.standard_normal((n, h)).astype(np.float32) * noise
        n_stress  = int(h * 0.75)                    # 12 stress-sensitive units
        for col in range(h):
            d = 1.0 if col < n_stress else -0.5       # calm units get inverted signal
            self.W_i[:self.input_size, col] += (STRESS_WEIGHTS * d * 0.55).astype(np.float32)
        self.b_i = np.full(h, -0.3, dtype=np.float32)

        # Cell candidate  ← direction of stress signal
        self.W_g = rng.standard_normal((n, h)).astype(np.float32) * noise
        for col in range(h):
            d = 1.0 if col < n_stress else -0.5
            self.W_g[:self.input_size, col] += (STRESS_WEIGHTS * d * 0.85).astype(np.float32)
        self.b_g = np.zeros(h, dtype=np.float32)

        # Output gate  ← moderate, slightly open
        self.W_o = rng.standard_normal((n, h)).astype(np.float32) * noise
        self.b_o = np.full(h, 0.25, dtype=np.float32)

        # Dense projection  h(16) → scalar stress score
        w_raw = np.array([1.0] * n_stress + [-0.5] * (h - n_stress), dtype=np.float32)
        self.W_out = w_raw / (np.linalg.norm(w_raw) + 1e-8)
        self.b_out = np.float32(0.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _step(self, x):
        xh  = np.concatenate([x, self.h])
        f   = _sigmoid(xh @ self.W_f + self.b_f)
        i   = _sigmoid(xh @ self.W_i + self.b_i)
        g   = _tanh(   xh @ self.W_g + self.b_g)
        o   = _sigmoid(xh @ self.W_o + self.b_o)
        self.c = f * self.c + i * g
        self.h = o * _tanh(self.c)

    def _normalise(self, probs):
        p = np.asarray(probs, dtype=np.float32).flatten()[:7]
        return p / (p.sum() + 1e-8)

    def _rule_score(self):
        if not self.buffer:
            return 50.0
        scores  = [
            float(np.clip(50.0 + 50.0 * float(STRESS_WEIGHTS @ x), 0.0, 100.0))
            for x in self.buffer
        ]
        # Recent frames weighted more
        weights = np.linspace(0.5, 1.0, len(scores))
        return float(np.average(scores, weights=weights))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self):
        """Reset LSTM hidden state and frame buffer."""
        self.h      = np.zeros(self.hidden_size, dtype=np.float32)
        self.c      = np.zeros(self.hidden_size, dtype=np.float32)
        self.buffer = []

    def update(self, probs):
        """Feed one CNN emotion probability vector (7-dim) into the LSTM."""
        x = self._normalise(probs)
        self.buffer.append(x.copy())
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self._step(x)

    def predict(self):
        """
        Temporal stress prediction (requires ≥1 update call).
        Returns
        -------
        score   : float, 0–100
        level   : 'low' | 'medium' | 'high'
        breakdown : dict with lstm_score, rule_score, per_emotion, window_len
        """
        raw        = float(self.h @ self.W_out) + float(self.b_out)
        lstm_score = float(_sigmoid(np.float32(raw))) * 100.0
        rule_score = self._rule_score()

        # Hybrid ensemble
        score = float(np.clip(0.6 * lstm_score + 0.4 * rule_score, 0.0, 100.0))
        level = "high" if score >= STRESS_HIGH else "medium" if score >= STRESS_MEDIUM else "low"

        last = self.buffer[-1] if self.buffer else np.zeros(7, dtype=np.float32)
        per_emotion = {EMOTIONS[i]: float(last[i]) for i in range(7)}

        return score, level, {
            "lstm_score" : round(lstm_score, 1),
            "rule_score" : round(rule_score, 1),
            "per_emotion": per_emotion,
            "window_len" : len(self.buffer),
        }

    def instant_stress(self, probs):
        """
        Single-shot stress from one CNN output (no temporal context).
        Returns same tuple as predict().
        """
        x     = self._normalise(probs)
        raw   = float(STRESS_WEIGHTS @ x)
        score = float(np.clip(50.0 + 50.0 * raw, 0.0, 100.0))
        level = "high" if score >= STRESS_HIGH else "medium" if score >= STRESS_MEDIUM else "low"
        dominant = EMOTIONS[int(np.argmax(x))]
        return score, level, {
            "lstm_score" : None,
            "rule_score" : round(score, 1),
            "per_emotion": {EMOTIONS[i]: float(x[i]) for i in range(7)},
            "dominant"   : dominant,
            "window_len" : 0,
        }
