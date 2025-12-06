import cv2
import mediapipe as mp
import time
import csv
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_FILE = "latency_log.csv"
MAX_FRAMES_TO_RECORD = 300  # Record for about 10 seconds (at 30fps)

# ==========================================
# 1. SETUP LOGIC (Your Updated Rules)
# ==========================================
mp_holistic = mp.solutions.holistic

def landmark_xy(landmarks, idx):
    lm = landmarks[idx]
    return lm.x, lm.y, lm.visibility

def center(a, b):
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)

def visible(*vs, thr=0.5):
    return all(v >= thr for v in vs)

def detect_rule_based(pose_landmarks):
    # --- START LOGIC TIMER ---
    t_start = time.perf_counter()
    
    if not pose_landmarks:
        return "neutral", (time.perf_counter() - t_start) * 1000

    lms = pose_landmarks.landmark
    
    # UPDATED THRESHOLDS (From your calibration)
    LEAN_THRESH = 0.0316
    HANDS_ABOVE_SHOULDERS_DELTA = -0.4119
    CROUCH_TORSO_RATIO = 0.80 

    NOSE = mp_holistic.PoseLandmark.NOSE
    L_SHO = mp_holistic.PoseLandmark.LEFT_SHOULDER
    R_SHO = mp_holistic.PoseLandmark.RIGHT_SHOULDER
    L_HIP = mp_holistic.PoseLandmark.LEFT_HIP
    R_HIP = mp_holistic.PoseLandmark.RIGHT_HIP
    L_WRIST = mp_holistic.PoseLandmark.LEFT_WRIST
    R_WRIST = mp_holistic.PoseLandmark.RIGHT_WRIST

    nose = landmark_xy(lms, NOSE)
    lsho = landmark_xy(lms, L_SHO)
    rsho = landmark_xy(lms, R_SHO)
    lhip = landmark_xy(lms, L_HIP)
    rhip = landmark_xy(lms, R_HIP)
    lwri = landmark_xy(lms, L_WRIST)
    rwri = landmark_xy(lms, R_WRIST)

    gesture = "neutral"

    if visible(nose[2], lsho[2], rsho[2], lhip[2], rhip[2], lwri[2], rwri[2]):
        sh_cx, sh_cy = center(lsho, rsho)
        hip_cx, hip_cy = center(lhip, rhip)
        dx = sh_cx - hip_cx
        torso = abs(nose[1] - hip_cy)
        shoulder_y = (lsho[1] + rsho[1]) * 0.5
        wrist_left_y = lwri[1]
        wrist_right_y = rwri[1]

        if wrist_left_y < (shoulder_y - HANDS_ABOVE_SHOULDERS_DELTA) and \
           wrist_right_y < (shoulder_y - HANDS_ABOVE_SHOULDERS_DELTA):
            gesture = "jump"
        elif torso < CROUCH_TORSO_RATIO:
            gesture = "duck"
        elif dx <= -LEAN_THRESH:
            gesture = "left"
        elif dx >= LEAN_THRESH:
            gesture = "right"
            
    # --- END LOGIC TIMER ---
    t_end = time.perf_counter()
    duration_ms = (t_end - t_start) * 1000
    return gesture, duration_ms

def simulate_keyboard_input(gesture):
    """
    Simulates the time it takes to press a key.
    We don't actually press it here to avoid closing windows, 
    but we execute the logic branching.
    """
    t_start = time.perf_counter()
    
    if gesture == "jump":
        pass # keyboard.press('space')
    elif gesture == "duck":
        pass # keyboard.press('down')
    elif gesture == "left":
        pass # keyboard.press('left')
    elif gesture == "right":
        pass # keyboard.press('right')
    else:
        pass 

    t_end = time.perf_counter()
    return (t_end - t_start) * 1000

# ==========================================
# 2. MAIN MEASUREMENT LOOP
# ==========================================
def measure_performance():
    cap = cv2.VideoCapture(0) # 0 for Webcam
    
    # Data storage
    records = []
    
    print(f"Starting Measurement for {MAX_FRAMES_TO_RECORD} frames...")
    print("Perform gestures in front of the camera!")
    time.sleep(2)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:
        
        frame_idx = 0
        while cap.isOpened() and frame_idx < MAX_FRAMES_TO_RECORD:
            
            # 1. READ FRAME
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret: break
            t1 = time.perf_counter() # Time after capture
            
            # 2. MEDIAPIPE PROCESSING
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            t2 = time.perf_counter() # Time after AI model
            
            # 3. RULE-BASED LOGIC
            gesture, logic_duration_ms = detect_rule_based(results.pose_landmarks)
            
            # 4. KEYBOARD INPUT SIMULATION
            input_duration_ms = simulate_keyboard_input(gesture)
            
            t3 = time.perf_counter() # Total end time
            
            # --- CALCULATIONS ---
            camera_ms = (t1 - t0) * 1000
            model_ms  = (t2 - t1) * 1000
            total_processing_ms = (t3 - t1) * 1000 # AI + Logic + Input (The "System Latency")
            total_fps_est = 1000 / total_processing_ms if total_processing_ms > 0 else 0
            
            print(f"Frame {frame_idx}: Latency={total_processing_ms:.2f}ms | Gesture={gesture}")
            
            records.append([frame_idx, camera_ms, model_ms, logic_duration_ms, input_duration_ms, total_processing_ms, gesture])
            
            # Show Feed
            cv2.putText(frame, f"Lat: {total_processing_ms:.1f}ms", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Latency Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # ==========================================
    # 3. SAVE & REPORT
    # ==========================================
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Camera_Read_ms", "MediaPipe_ms", "Logic_ms", "Input_ms", "Total_System_Latency_ms", "Gesture"])
        writer.writerows(records)

    # Calculate Averages
    data = np.array(records)[:, 1:6].astype(float) # Columns 1 to 5
    avg_camera = np.mean(data[:, 0])
    avg_model = np.mean(data[:, 1])
    avg_logic = np.mean(data[:, 2])
    avg_input = np.mean(data[:, 3])
    avg_total = np.mean(data[:, 4])

    print("\n" + "="*40)
    print("       LATENCY PERFORMANCE REPORT       ")
    print("="*40)
    print(f"Total Frames Recorded: {frame_idx}")
    print(f"Average Total Latency: {avg_total:.2f} ms")
    print("-" * 30)
    print(f"Breakdown:")
    print(f"1. Camera Read:        {avg_camera:.2f} ms (Hardware limit)")
    print(f"2. AI Model (Pipe):    {avg_model:.2f} ms (Biggest Bottleneck)")
    print(f"3. Logic Rules:        {avg_logic:.4f} ms (Very Fast)")
    print(f"4. Input Trigger:      {avg_input:.4f} ms (Negligible)")
    print("-" * 30)
    print(f"Estimated Max FPS:     {1000/avg_total:.1f} FPS")
    print(f"Data saved to: {OUTPUT_FILE}")
    print("="*40)

if __name__ == "__main__":
    measure_performance()