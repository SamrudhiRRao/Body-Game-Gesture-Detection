import time
from collections import deque, Counter

import cv2
import mediapipe as mp
import numpy as np
import joblib
import pydirectinput

MODEL_PATH = "Train Model/karate_rf_model.pkl"
ENCODER_PATH = "Train Model/label_encoder.pkl"


SEQUENCE_LENGTH = 60  # FRAMES_PER_CLIP

# Gesture - Game key mapping
# Move: WASD, Attacks: J K L I, Combo: U
GESTURE_TO_ACTION = {
    "neutral": None,

    "move_left":  ("hold", "a"),
    "move_right": ("hold", "d"),

    "jump":   ("tap", "w"),
    "crouch": ("hold", "s"),

    "low_punch":   ("tap", "j"),
    "high_punch":  ("tap", "j"),
    "high_kick":   ("tap", "k"),
    "strong_kick": ("tap", "i"),

    "hit_combo": ("tap", "u"),
}

# Cooldowns to avoid spamming
COOLDOWN = {
    "low_punch": 0.18,
    "high_punch": 0.18,
    "high_kick": 0.22,
    "strong_kick": 0.35,
    "hit_combo": 0.60,
    "jump": 0.25,
}
DEFAULT_TAP_COOLDOWN = 0.20

# Smoothing / stability
SMOOTH_WINDOW = 7
MIN_STABLE_COUNT = 5

model = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
drawer = mp.solutions.drawing_utils

def tap_key(k: str):
    pydirectinput.press(k)

def hold_key(k: str):
    pydirectinput.keyDown(k)

def release_key(k: str):
    pydirectinput.keyUp(k)

held_key = None
last_tap_time = 0.0

pred_hist = deque(maxlen=SMOOTH_WINDOW)

def get_stable_label():
    if len(pred_hist) < SMOOTH_WINDOW:
        return None
    c = Counter(pred_hist)
    lab, cnt = c.most_common(1)[0]
    return lab if cnt >= MIN_STABLE_COUNT else None

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

buffer = deque(maxlen=SEQUENCE_LENGTH)
stable_label = "neutral"

print("\nHOW TO USE:")
print("1) Open https://poki.com/en/g/karate-fighter and CLICK inside the game (important).")
print("2) Run this script. Keep the browser focused so it receives keys.")
print("3) Press 'q' in the webcam window to quit.\n")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        raw_label = None

        if res.pose_landmarks:
            lm_list = [[lm.x, lm.y, lm.z] for lm in res.pose_landmarks.landmark]
            buffer.append(lm_list)

            drawer.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if len(buffer) == SEQUENCE_LENGTH:
                sample = np.array(buffer, dtype=np.float32)     # (30, 33, 3)
                X = sample.flatten().reshape(1, -1)             # (1, 2970)

                pred = model.predict(X)
                raw_label = le.inverse_transform(pred)[0]
                pred_hist.append(raw_label)

                st = get_stable_label()
                if st is not None:
                    stable_label = st
        else:
            # If no pose detected, drift toward neutral (prevents stuck movement)
            pred_hist.append("neutral")
            st = get_stable_label()
            if st is not None:
                stable_label = st

        action = GESTURE_TO_ACTION.get(stable_label, None)
        now = time.time()

        if action is None or action[0] != "hold":
            if held_key is not None:
                release_key(held_key)
                held_key = None

        if action is not None:
            kind, key = action
            if kind == "hold":
                if held_key != key:
                    if held_key is not None:
                        release_key(held_key)
                    hold_key(key)
                    held_key = key
            elif kind == "tap":
                cooldown = COOLDOWN.get(stable_label, DEFAULT_TAP_COOLDOWN)
                if now - last_tap_time >= cooldown:
                    tap_key(key)
                    last_tap_time = now

        cv2.putText(frame, f"raw: {raw_label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"stable: {stable_label}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow("Karate Fighter Gesture Control (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    if held_key is not None:
        release_key(held_key)
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
