import cv2
import mediapipe as mp
import numpy as np
import joblib
import pydirectinput
from collections import deque, Counter

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "karate_model_new.pkl"
CONFIDENCE_THRESHOLD = 0.5

# The same 12 landmarks used in training
BODY_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# KEY MAPPINGS (Adjust 'neutral' or gesture names to match your CSV labels exactly)
KEY_MAP = {
    "neutral": None,      # Do nothing
    "idle": None,         # Handle 'idle' if you used that name
    
    "low_punch": "l",
    "high_punch": "i",    # Both punches hit 'J'
    
    "strong_kick": "k",
    "high_kick": "k",     # Both kicks hit 'K'
    
    "hit_combo": "u",     # Special move
    
    "jump": "w",
    "crouch": "s",
    "move_left": "a",
    "move_right": "d"
}

# ==========================================
# SETUP
# ==========================================
print(f"Loading model from {MODEL_PATH}...")
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"Error: Could not find {MODEL_PATH}. Run optimized_trainer.py first!")
    exit()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Window Setup
cv2.namedWindow("Karate Game Controller", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Karate Game Controller", 800, 600)

# Smoothing Buffer (Low Latency)
# We only look at the last 2-3 frames to stop flickering
history = deque(maxlen=3) 
current_key = None


def press_key(gesture):
    global current_key
    target_key = KEY_MAP.get(gesture)
    
    # Only change state if the key is different
    if target_key != current_key:
        if current_key: 
            pydirectinput.keyUp(current_key) # Release old key
        if target_key: 
            pydirectinput.keyDown(target_key) # Press new key
        
        current_key = target_key

print("Game Controller Active. Switch to your game window!")
print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # 1. Image Processing
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    final_gesture = "neutral"

    # 2. Extract Features
    if results.pose_landmarks:
        # Draw skeleton for feedback
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        row = []
        # EXTRACT ONLY THE 12 OPTIMIZED POINTS
        for idx in BODY_INDICES:
            lm = results.pose_landmarks.landmark[idx]
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
        
        # 3. Predict
        try:
            # Predict expects a 2D array, so we wrap row in []
            prediction = model.predict([row])[0]
            probs = model.predict_proba([row])[0]
            confidence = np.max(probs)

            # 4. Smoothing Logic
            if confidence > CONFIDENCE_THRESHOLD:
                history.append(prediction)
                # Take the most common gesture in the last 3 frames
                final_gesture = Counter(history).most_common(1)[0][0]
                
                # Visual Confidence Bar
                bar_width = int(confidence * 200)
                cv2.rectangle(frame, (20, 100), (20 + bar_width, 115), (0, 255, 0), -1)
                
        except Exception as e:
            print(f"Prediction Error: {e}")

    # 5. Execute Game Command
    press_key(final_gesture)

    # 6. UI Display
    color = (0, 255, 0) if final_gesture != "neutral" else (200, 200, 200)
    
    # Main Gesture Text
    cv2.putText(frame, f"ACTION: {final_gesture.upper()}", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Karate Game Controller", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
if current_key: 
    pydirectinput.keyUp(current_key)
cap.release()
cv2.destroyAllWindows()