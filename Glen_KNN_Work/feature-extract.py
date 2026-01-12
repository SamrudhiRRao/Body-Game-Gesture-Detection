import cv2
import mediapipe as mp
import numpy as np
import os
import time
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
GESTURES = [
    "jump", "crouch", "move_left", "move_right",
    "quick_punch", "kick",
    "attack","idle"
]

# 3 seconds * ~30 frames per second = 90 frames
FRAMES_PER_CLIP = 90 
CLIPS_PER_GESTURE = 10

OUTPUT_DIR = "knn_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create folders
for g in GESTURES:
    os.makedirs(os.path.join(OUTPUT_DIR, g), exist_ok=True)

# MediaPipe Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
# === ADD THESE LINES ===
cv2.namedWindow("Recording", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Recording", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# =======================

def draw_text_centered(image, text, y_pos, scale=1.0, color=(0, 255, 0), thickness=2):
    """Helper to center text on the screen"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    cv2.putText(image, text, (text_x, y_pos), font, scale, color, thickness)

def capture_clip(label, clip_no):
    keypoints_list = []
    print(f"\nPrepare for {label} clip {clip_no+1}")

    # ==========================================
    # 1. VISUAL COUNTDOWN (3 Seconds)
    # ==========================================
    countdown_duration = 3
    start_countdown = time.time()
    
    # We loop until 3 seconds have passed
    while True:
        elapsed = time.time() - start_countdown
        remaining = countdown_duration - int(elapsed)
        
        if remaining <= 0:
            break
            
        ret, frame = cap.read()
        if not ret:
            break
            
        # UI: Draw Countdown
        draw_text_centered(frame, f"NEXT: {label.upper()}", 100, 1.5, (255, 0, 0), 3)
        draw_text_centered(frame, str(remaining), 250, 4.0, (0, 0, 255), 5)
        
        cv2.imshow("Recording", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            return False # Signal to stop

    # ==========================================
    # 2. RECORDING PHASE (3 Seconds / 90 Frames)
    # ==========================================
    print(f"Recording {label}...")
    frame_count = 0
    
    while frame_count < FRAMES_PER_CLIP:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            kps = [[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark]
            keypoints_list.append(kps)
            
            # Optional: Draw skeleton for feedback
            mp.solutions.drawing_utils.draw_landmarks(
                frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
            # If body lost, append zeros (33 landmarks * 4 values [x,y,z,v])
            keypoints_list.append(np.zeros((33,4)))

        # UI: Recording Status
        cv2.putText(frame, f"RECORDING: {label}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}/{FRAMES_PER_CLIP}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Visual Progress Bar
        bar_width = int((frame_count / FRAMES_PER_CLIP) * 600)
        cv2.rectangle(frame, (20, 440), (20 + bar_width, 460), (0, 255, 0), -1)

        cv2.imshow("Recording", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

    # Save Data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(OUTPUT_DIR, label, f"{timestamp}.npy")
    np.save(save_path, np.array(keypoints_list))
    print(f"Saved {save_path}")
    
    return True

# ==========================================
# MAIN LOOP
# ==========================================
def main():
    running = True
    for g in GESTURES:
        if not running: break
        
        for clip_no in range(CLIPS_PER_GESTURE):
            if not running: break
            
            # Capture clip, if returns False, user pressed 'q'
            running = capture_clip(g, clip_no)

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection complete.")

if __name__ == "__main__":
    main()