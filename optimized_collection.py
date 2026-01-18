import cv2
import mediapipe as mp
import pandas as pd
import os
import time

# --- CONFIGURATION ---
OUTPUT_FILE = "karate_optimized_data2.csv"
GESTURES = ["neutral", "low_punch", "high_punch", "crouch", "jump", "strong_kick", "hit_combo", "move_left", "move_right", "high_kick"]

COUNTDOWN_SECONDS = 3   # Time to get ready
RECORD_SECONDS = 10     # Time to actually record data 



# We only keep these 12 Landmarks (Body Core)
BODY_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Setup MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Create CSV if it doesn't exist
if not os.path.exists(OUTPUT_FILE):
    headers = ['label'] + [f'v{i}' for i in range(len(BODY_INDICES) * 4)]
    pd.DataFrame(columns=headers).to_csv(OUTPUT_FILE, index=False)

cap = cv2.VideoCapture(0)

# =================================================================
# 1. WINDOW RESIZING (Change numbers here to adjust size)
# =================================================================
cv2.namedWindow("Optimized Collector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Optimized Collector", 1280, 720) 
# =================================================================

current_idx = 0
is_recording = False
countdown_active = False
state_start_time = 0 

def draw_text_with_bg(img, text, x, y, font_scale=0.7, thickness=2, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (x - 5, y - text_h - 5), (x + text_w + 5, y + 10), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

print("--- OPTIMIZED COLLECTION ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    gesture_name = GESTURES[current_idx]
    
    # UI Info
    try: samples = len(pd.read_csv(OUTPUT_FILE))
    except: samples = 0

    draw_text_with_bg(frame, f"TARGET: {gesture_name.upper()}", 20, 50, 1.0, 2, (0, 255, 0))
    draw_text_with_bg(frame, f"Samples: {samples}", 20, 90, 0.7, 2, (255, 255, 0))
    draw_text_with_bg(frame, "[N]: Next | [SPACE]: Start Session | [Q]: Quit", 20, frame.shape[0]-30, 0.6)

    # ==========================
    # LOGIC FLOW
    # ==========================
    
    # 1. GET READY COUNTDOWN (3s)
    if countdown_active:
        elapsed = time.time() - state_start_time
        remaining = COUNTDOWN_SECONDS - elapsed
        
        if remaining > 0:
            cx, cy = frame.shape[1]//2, frame.shape[0]//2
            draw_text_with_bg(frame, f"GET READY: {int(remaining)+1}", cx-100, cy, 2.5, 5, (0, 255, 255))
        else:
            # Transition to Recording
            countdown_active = False
            is_recording = True
            state_start_time = time.time() # Reset timer for recording phase

    # 2. RECORDING PHASE (10s)
    elif is_recording:
        elapsed = time.time() - state_start_time
        remaining = RECORD_SECONDS - elapsed

        if remaining > 0:
            # Draw Timer
            cx = frame.shape[1]//2
            draw_text_with_bg(frame, f"RECORDING: {remaining:.1f}s", cx-150, 100, 1.5, 3, (0, 0, 255), (255, 255, 255))
            
            # Save Data
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                row = [gesture_name]
                for idx in BODY_INDICES:
                    lm = results.pose_landmarks.landmark[idx]
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])
                
                pd.DataFrame([row]).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
        else:
            # Time is up
            is_recording = False

    # 3. IDLE PHASE
    else:
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Optimized Collector", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('n') and not is_recording and not countdown_active: 
        current_idx = (current_idx + 1) % len(GESTURES)
    elif key == 32: # SPACE BAR
        if not is_recording and not countdown_active:
            # Start the sequence
            countdown_active = True
            state_start_time = time.time()

cap.release()
cv2.destroyAllWindows()