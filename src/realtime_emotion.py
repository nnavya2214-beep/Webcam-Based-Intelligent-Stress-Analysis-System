"""
src/realtime_emotion.py
Real-time emotion recognition from webcam feed
FIXED VERSION - Works from any directory
"""

import cv2
import numpy as np
import os
import sys
import time

# Fix path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Now import TensorFlow (after path fix)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

# Emotion labels (inline to avoid import issues)
EMOTION_LIST = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def get_emotion_label(index):
    """Get emotion label from index"""
    if 0 <= index < len(EMOTION_LIST):
        return EMOTION_LIST[index]
    return 'unknown'


def _stress_score_and_level(probs):
    """Stress from CNN probs + rule-based level. Uses utils.stress if available."""
    try:
        from utils.stress import stress_score_from_cnn, rule_based_stress_level
        score = stress_score_from_cnn(probs)
        level = rule_based_stress_level(score)
        return score, level
    except ImportError:
        # Fallback: simple weighted sum (same order as EMOTION_LIST)
        w = np.array([1.0, 0.9, 1.0, -0.8, 0.95, -0.2, 0.1])
        p = np.asarray(probs).flatten()
        if len(p) != 7:
            return 50.0, "low"
        raw = np.clip(np.dot(w, p), -1.0, 1.0)
        score = float(np.clip(50.0 + 50.0 * raw, 0.0, 100.0))
        level = "high" if score >= 65 else "medium" if score >= 40 else "low"
        return score, level

class FaceDetector:
    """Simple face detector using Haar Cascade"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces
    
    def extract_face_roi(self, frame, face_coords, target_size=(48, 48)):
        """Extract and preprocess face region"""
        x, y, w, h = face_coords
        
        # Bounds checking
        h_img, w_img = frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        
        # Extract face
        face_roi = frame[y:y+h, x:x+w]
        
        # Convert to grayscale
        if len(face_roi.shape) == 3:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize
        face_roi = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_AREA)
        
        return face_roi
    
    def preprocess_for_model(self, face_image):
        """Preprocess face for model input"""
        face_normalized = face_image.astype('float32') / 255.0
        face_reshaped = face_normalized.reshape(1, 48, 48, 1)
        return face_reshaped

class RealtimeEmotionRecognizer:
    """Real-time emotion recognition system"""
    
    def __init__(self, model_path):
        print("[INFO] Loading emotion recognition model...")
        self.model = load_model(model_path)
        print("[INFO] Model loaded successfully (66.43% accuracy)")
        
        print("[INFO] Initializing face detector...")
        self.face_detector = FaceDetector()
        print("[INFO] Face detector initialized")
        
        # Emotion colors (BGR format)
        self.emotion_colors = {
            'angry': (0, 0, 255),      # Red
            'disgust': (0, 128, 0),    # Dark Green
            'fear': (255, 0, 255),     # Magenta
            'happy': (0, 255, 255),    # Yellow
            'sad': (255, 0, 0),        # Blue
            'surprise': (255, 255, 0), # Cyan
            'neutral': (128, 128, 128) # Gray
        }
    
    def predict_emotion(self, face_image):
        """Predict emotion from face image. Returns emotion, confidence, probs, stress_score, stress_level."""
        predictions = self.model.predict(face_image, verbose=0)
        probs = predictions[0]
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]
        emotion = get_emotion_label(predicted_class)
        stress_score, stress_level = _stress_score_and_level(probs)
        return emotion, confidence, probs, stress_score, stress_level
    
    def draw_emotion_info(self, frame, face_coords, emotion, confidence, probabilities, stress_score=None, stress_level=None):
        """Draw emotion information on frame. If stress_level is 'high', draw alert."""
        x, y, w, h = face_coords
        color = self.emotion_colors.get(emotion, (0, 255, 0))
        
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Draw label
        label = f"{emotion.capitalize()}: {confidence*100:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        (text_width, text_height), _ = cv2.getTextSize(label, font, 0.8, 2)
        
        cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width + 5, y), color, -1)
        cv2.putText(frame, label, (x, y - 5), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # High-stress alert (CNN + rule-based)
        if stress_level == "high":
            alert_y = y + h + 28
            cv2.rectangle(frame, (x, y + h + 2), (x + 220, alert_y), (0, 0, 255), 2)
            cv2.rectangle(frame, (x + 1, y + h + 3), (x + 219, alert_y - 1), (0, 0, 200), -1)
            cv2.putText(frame, "HIGH STRESS - Take a break", (x + 6, alert_y - 10),
                        font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        elif stress_score is not None and stress_level:
            stress_text = f"Stress: {stress_score:.0f} ({stress_level})"
            cv2.putText(frame, stress_text, (x, y + h + 20), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Draw probability bars
        self._draw_emotion_bars(frame, probabilities)
        
        return frame
    
    def _draw_emotion_bars(self, frame, probabilities, bar_height=18):
        """Draw emotion probability bars"""
        h, w = frame.shape[:2]
        start_x = 10
        start_y = h - len(EMOTION_LIST) * (bar_height + 4) - 30
        bar_width = 150
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, start_y - 25), (bar_width + 120, h - 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Title
        cv2.putText(frame, "Emotion Probabilities", (start_x, start_y - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Bars
        for i, (emotion, prob) in enumerate(zip(EMOTION_LIST, probabilities)):
            y_pos = start_y + i * (bar_height + 4)
            
            # Label
            cv2.putText(frame, emotion.capitalize()[:7], (start_x, y_pos + 14),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Bar
            bar_length = int(prob * bar_width)
            color = self.emotion_colors.get(emotion, (0, 255, 0))
            cv2.rectangle(frame, (start_x + 55, y_pos + 2),
                         (start_x + 55 + bar_length, y_pos + bar_height - 2), color, -1)
            
            # Percentage
            cv2.putText(frame, f"{prob*100:.1f}%", (start_x + bar_width + 60, y_pos + 14),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    
    def run(self, camera_index=0):
        """Run real-time emotion recognition"""
        print("\n" + "="*60)
        print("  REAL-TIME EMOTION RECOGNITION (66.43% Accuracy)")
        print("="*60)
        print("  Press 'q' to quit")
        print("  Press 's' to save screenshot")
        print("="*60 + "\n")
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("[INFO] Webcam opened. Starting recognition...")
        
        frame_count = 0
        fps = 0
        fps_start_time = time.time()
        screenshot_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            
            # Process each face
            for face in faces:
                face_roi = self.face_detector.extract_face_roi(frame, face)
                face_preprocessed = self.face_detector.preprocess_for_model(face_roi)
                
                emotion, confidence, probabilities, stress_score, stress_level = self.predict_emotion(face_preprocessed)
                self.draw_emotion_info(frame, face, emotion, confidence, probabilities, stress_score, stress_level)
            
            # FPS calculation
            frame_count += 1
            if frame_count >= 10:
                fps = frame_count / (time.time() - fps_start_time)
                frame_count = 0
                fps_start_time = time.time()
            
            # Display info
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 100, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces: {len(faces)}", (frame.shape[1] - 100, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Emotion Recognition - 66.43% Accuracy', frame)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_count += 1
                os.makedirs('screenshots', exist_ok=True)
                filename = f'screenshots/emotion_{screenshot_count}.jpg'
                cv2.imwrite(filename, frame)
                print(f"[INFO] Screenshot saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Emotion recognition stopped")

def main():
    """Main function"""
    # Find model
    possible_paths = [
        'models/best.h5',
        '../models/best.h5',
        'models/final.h5',
        '../models/final.h5'
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("\n[ERROR] Model not found!")
        print("Looked in:", possible_paths)
        print("\nPlease make sure models/best.h5 exists")
        return
    
    print(f"[INFO] Found model at: {model_path}")
    
    # Run
    recognizer = RealtimeEmotionRecognizer(model_path=model_path)
    recognizer.run(camera_index=0)

if __name__ == "__main__":
    main()