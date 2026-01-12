import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from collections import deque

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "knn_gesture_model.pkl"
FRAMES_TO_COLLECT = 90  # Must match training data
PREDICTION_INTERVAL = 10 # How often to predict (every 10 frames)

# MediaPipe Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load Model
print(f"Loading model from {MODEL_PATH}...")
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: Model file not found. Run knn_trainer.py first to save the model.")
    exit()

# Buffer to store the last 90 frames of keypoints
# Deque automatically removes old items when full
data_buffer = deque(maxlen=FRAMES_TO_COLLECT)

cap = cv2.VideoCapture(0)

# Fullscreen setup
cv2.namedWindow("Live Gesture Recognition", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Live Gesture Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

current_prediction = "Waiting for data..."
confidence_text = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Process Frame with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    # 2. Extract Keypoints
    if result.pose_landmarks:
        # Draw skeleton
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Extract data (33 landmarks * 4 values = 132 features)
        kps = [[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark]
        flattened_kps = np.array(kps).flatten()
        
        # Add to buffer
        data_buffer.append(flattened_kps)
    else:
        # If no body detected, pad with zeros to keep timeline moving
        data_buffer.append(np.zeros(33 * 4))

    # 3. Predict Gesture
    # Only predict if we have enough frames (90)
    if len(data_buffer) == FRAMES_TO_COLLECT:
        # Convert buffer to a single flat array (what KNN expects)
        # Shape becomes (1, 90*132) -> (1, 11880)
        input_vector = np.array(data_buffer).flatten().reshape(1, -1)
        
        try:
            prediction = model.predict(input_vector)[0]
            
            # Optional: Get probability if model supports it
            probs = model.predict_proba(input_vector)[0]
            confidence = np.max(probs) * 100
            
            current_prediction = prediction
            confidence_text = f"Conf: {confidence:.1f}%"
            
            # Color logic
            color = (0, 255, 0) # Green for high confidence
            if confidence < 60:
                color = (0, 165, 255) # Orange for unsure
                current_prediction += " (?)"
                
        except Exception as e:
            print(f"Prediction Error: {e}")

    # 4. Display UI
    # Progress Bar (how much data we have)
    buffer_len = len(data_buffer)
    bar_width = int((buffer_len / FRAMES_TO_COLLECT) * 200)
    cv2.rectangle(frame, (20, 400), (20 + bar_width, 415), (255, 255, 0), -1)
    
    # Text
    cv2.putText(frame, f"GESTURE: {current_prediction}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(frame, confidence_text, (20, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(frame, f"Buffer: {buffer_len}/{FRAMES_TO_COLLECT}", (20, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    cv2.imshow("Live Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()