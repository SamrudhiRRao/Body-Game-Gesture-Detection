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
FRAMES_TO_COLLECT = 90  
PREDICTION_INTERVAL = 10 

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

# Buffer
data_buffer = deque(maxlen=FRAMES_TO_COLLECT)

cap = cv2.VideoCapture(0)

# Fullscreen setup
cv2.namedWindow("Live Gesture Recognition", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Live Gesture Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

current_prediction = "Waiting..."
confidence_text = ""

# TIMING VARIABLES
inference_ms = 0
process_ms = 0
fps = 0
prev_frame_time = 0

while cap.isOpened():
    # Start timer for TOTAL loop (FPS)
    new_frame_time = time.perf_counter()
    
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Process Frame with MediaPipe (Measure Time)
    process_start = time.perf_counter()
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)
    
    process_end = time.perf_counter()
    process_ms = (process_end - process_start) * 1000  # Convert to ms

    # 2. Extract Keypoints
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        kps = [[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark]
        flattened_kps = np.array(kps).flatten()
        data_buffer.append(flattened_kps)
    else:
        data_buffer.append(np.zeros(33 * 4))

    # 3. Predict Gesture (Measure Time)
    if len(data_buffer) == FRAMES_TO_COLLECT:
        input_vector = np.array(data_buffer).flatten().reshape(1, -1)
        
        # Start Inference Timer
        infer_start = time.perf_counter()
        
        try:
            prediction = model.predict(input_vector)[0]
            
            # Optional: Get probability
            probs = model.predict_proba(input_vector)[0]
            confidence = np.max(probs) * 100
            
            current_prediction = prediction
            confidence_text = f"Conf: {confidence:.1f}%"
            
        except Exception as e:
            print(f"Prediction Error: {e}")
            
        # End Inference Timer
        infer_end = time.perf_counter()
        inference_ms = (infer_end - infer_start) * 1000

    # 4. Display UI & Metrics
    
    # Calculate FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    
    # -- Draw Metrics Box (Top Right) --
    cv2.rectangle(frame, (frame.shape[1] - 220, 0), (frame.shape[1], 110), (0, 0, 0), -1)
    
    # Display FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 200, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display MediaPipe Time
    cv2.putText(frame, f"Process: {process_ms:.1f}ms", (frame.shape[1] - 200, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # Display KNN Time
    cv2.putText(frame, f"Inference: {inference_ms:.1f}ms", (frame.shape[1] - 200, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)


    # -- Draw Main Info --
    # Progress Bar
    buffer_len = len(data_buffer)
    bar_width = int((buffer_len / FRAMES_TO_COLLECT) * 200)
    cv2.rectangle(frame, (20, 400), (20 + bar_width, 415), (255, 255, 0), -1)
    
    cv2.putText(frame, f"GESTURE: {current_prediction}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(frame, confidence_text, (20, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    cv2.imshow("Live Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()