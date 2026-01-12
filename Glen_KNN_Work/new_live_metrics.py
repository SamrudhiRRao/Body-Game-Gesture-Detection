import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from collections import deque

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "optimized_karate_model.pkl"
CONFIDENCE_THRESHOLD = 0.5

# The 12 Core Landmarks
BODY_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# DISPLAY MAPPING (Gesture -> Key)
KEY_MAP_DISPLAY = {
    "neutral": "",
    "idle": "",
    "low_punch": "[L]",
    "high_punch": "[I]",
    "strong_kick": "[K]",
    "high_kick": "[K]",
    "hit_combo": "[U]",
    "jump": "[W]",
    "crouch": "[S]",
    "move_left": "[A]",
    "move_right": "[D]"
}

# ==========================================
# SETUP
# ==========================================
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Error: Could not find {MODEL_PATH}")
    exit()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cv2.namedWindow("Performance Metrics", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Performance Metrics", 1280, 720)

# Performance Timers
fps_history = deque(maxlen=30)
prev_frame_time = 0

def draw_hud(image, action, key, confidence, latency, fps):
    """Draws a futuristic, semi-transparent HUD"""
    h, w, _ = image.shape
    
    # 1. SIDEBAR BACKGROUND (Semi-transparent black)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (300, h), (10, 10, 10), -1)
    
    # 2. BOTTOM BAR (Action Display)
    cv2.rectangle(overlay, (0, h-100), (w, h), (0, 0, 0), -1)
    
    alpha = 0.6  # Transparency factor
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # --- METRICS PANEL (Left Side) ---
    # Title
    cv2.putText(image, "Gesture Control Karate GamePlay", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.line(image, (20, 60), (280, 60), (0, 255, 255), 1)

    # FPS
    color_fps = (0, 255, 0) if fps > 25 else (0, 165, 255)
    cv2.putText(image, f"FPS: {int(fps)}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_fps, 2)

    # Latency
    color_lat = (0, 255, 0) if latency < 30 else (0, 0, 255)
    cv2.putText(image, f"Latency: {latency:.1f}ms", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_lat, 2)

    # Confidence Text
    cv2.putText(image, f"Confidence: {int(confidence)}%", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Confidence Bar
    bar_x, bar_y, bar_w, bar_h = 20, 200, 200, 15
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    filled_w = int(bar_w * (confidence / 100))
    
    # Gradient Color for Bar (Red -> Yellow -> Green)
    if confidence > 80: bar_color = (0, 255, 0)
    elif confidence > 50: bar_color = (0, 255, 255)
    else: bar_color = (0, 0, 255)
    
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), bar_color, -1)

    # --- ACTION DISPLAY (Bottom Center) ---
    if action not in ["neutral", "idle"]:
        text = f"{action.upper()} {key}"
        color = (0, 255, 0)
    else:
        text = "IDLE"
        color = (100, 100, 100)
    
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(image, text, (text_x, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)

while cap.isOpened():
    # Loop Timer Start
    loop_start = time.perf_counter()
    
    ret, frame = cap.read()
    if not ret: break
    
    # Mirror frame
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 1. Processing Timer Start
    process_start = time.perf_counter()
    results = pose.process(rgb)
    
    final_gesture = "neutral"
    key_display = ""
    confidence_score = 0.0

    if results.pose_landmarks:
        # Dynamic Skeleton Color based on visibility
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))
        
        # Extract 12 Points
        row = []
        for idx in BODY_INDICES:
            lm = results.pose_landmarks.landmark[idx]
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
        
        # 2. Prediction
        try:
            probs = model.predict_proba([row])[0]
            confidence_score = np.max(probs) * 100
            
            if confidence_score > (CONFIDENCE_THRESHOLD * 100):
                final_gesture = model.predict([row])[0]
                key_display = KEY_MAP_DISPLAY.get(final_gesture, "")
                
        except Exception as e:
            pass

    # Timer Calculations
    process_end = time.perf_counter()
    latency_ms = (process_end - process_start) * 1000
    
    # FPS Calculation
    loop_end = time.perf_counter()
    fps = 1 / (loop_end - prev_frame_time) if (loop_end - prev_frame_time) > 0 else 0
    prev_frame_time = loop_end
    
    # Draw Interface
    draw_hud(frame, final_gesture, key_display, confidence_score, latency_ms, fps)

    cv2.imshow("Performance Metrics", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()