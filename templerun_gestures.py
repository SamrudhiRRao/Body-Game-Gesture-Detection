import time
import math
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

import csv

import pyautogui
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# Smoothing & debounce
EMA_ALPHA = 0.35
COOLDOWN_S = 0.8
EVENT_LOG = []
SHOW_DEBUG = True

# evaluation
EVAL_MODE = True  # False for normal play

# 5 times each gesture
SCRIPTED_GESTURES = (
    ["jump"] * 5 +
    ["duck"] * 5 +
    ["left"] * 5 +
    ["right"] * 5
)

STEP_DURATION = 2.0  # 2 seconds per gesture for now

GROUND_TRUTH_STEPS = []   # "index": int, "gesture": str, "t_start": float, "t_end": float | None

HANDS_ABOVE_SHOULDERS_DELTA = 0.02   # how far above shoulders wrists must be
CROUCH_TORSO_RATIO = 0.62            # nose-to-hip distance / standing reference below this duck
LEAN_THRESH = 0.06                    # shoulder_center - hip_center above this left/right
REQUIRE_BOTH_HANDS_FOR_JUMP = True


AUTO_CALIBRATE_STANDING = True
CALIB_FRAMES = 60

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def press(key):
    try:
        pyautogui.press(key)
    except Exception:
        pass

class EMA:
    def __init__(self, alpha, initial=None):
        self.alpha = alpha
        self.value = initial

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value

def landmark_xy(landmarks, idx):
    lm = landmarks[idx]
    return lm.x, lm.y, lm.visibility

def center(a, b):
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)

def visible(*vs, thr=0.5):
    return all(v >= thr for v in vs)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    last_trigger_time = {"jump": 0, "duck": 0, "left": 0, "right": 0}

    torso_ref_vals = deque(maxlen=CALIB_FRAMES)
    standing_torso_ref = None

    ema_dx = EMA(EMA_ALPHA)
    ema_torso = EMA(EMA_ALPHA)
    ema_wrist_left = EMA(EMA_ALPHA)
    ema_wrist_right = EMA(EMA_ALPHA)
    ema_shoulder_y = EMA(EMA_ALPHA)

    # MediaPipe Holistic
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        smooth_landmarks=True
    ) as holistic:

        eval_start_time = time.time()
        current_step_index = -1

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_ts = time.time()

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            res = holistic.process(rgb)
            pose = res.pose_landmarks

            dx_sm = 0.0
            torso_sm = None
            wrists_above = False

            if pose:
                lms = pose.landmark

                NOSE = mp_holistic.PoseLandmark.NOSE
                L_SHO = mp_holistic.PoseLandmark.LEFT_SHOULDER
                R_SHO = mp_holistic.PoseLandmark.RIGHT_SHOULDER
                L_HIP = mp_holistic.PoseLandmark.LEFT_HIP
                R_HIP = mp_holistic.PoseLandmark.RIGHT_HIP
                L_WRIST = mp_holistic.PoseLandmark.LEFT_WRIST
                R_WRIST = mp_holistic.PoseLandmark.RIGHT_WRIST

                # key points
                nose = landmark_xy(lms, NOSE)
                lsho = landmark_xy(lms, L_SHO)
                rsho = landmark_xy(lms, R_SHO)
                lhip = landmark_xy(lms, L_HIP)
                rhip = landmark_xy(lms, R_HIP)
                lwri = landmark_xy(lms, L_WRIST)
                rwri = landmark_xy(lms, R_WRIST)

                vis_ok = visible(nose[2], lsho[2], rsho[2], lhip[2], rhip[2], lwri[2], rwri[2], thr=0.5)

                if vis_ok:
                    sh_cx, sh_cy = center(lsho, rsho)
                    hip_cx, hip_cy = center(lhip, rhip)

                    dx = sh_cx - hip_cx  # lean left or right
                    torso = abs(nose[1] - hip_cy)
                    # wrists relative to shoulders
                    wrist_left_y = lwri[1]
                    wrist_right_y = rwri[1]
                    shoulder_y = (lsho[1] + rsho[1]) * 0.5

                    dx_sm = ema_dx.update(dx)
                    torso_sm = ema_torso.update(torso)
                    wl_sm = ema_wrist_left.update(wrist_left_y)
                    wr_sm = ema_wrist_right.update(wrist_right_y)
                    shy_sm = ema_shoulder_y.update(shoulder_y)

                    if AUTO_CALIBRATE_STANDING and (standing_torso_ref is None):
                        torso_ref_vals.append(torso_sm)
                        if len(torso_ref_vals) >= CALIB_FRAMES:
                            standing_torso_ref = float(np.median(torso_ref_vals))
                            standing_torso_ref = max(0.15, min(standing_torso_ref, 0.6))

                    now = time.time()

                    # ground truth
                    if EVAL_MODE:
                        t_eval = now - eval_start_time
                        total_steps = len(SCRIPTED_GESTURES)

                        step_index = int(t_eval // STEP_DURATION)

                        if step_index >= total_steps:
                            if current_step_index >= 0 and GROUND_TRUTH_STEPS:
                                if GROUND_TRUTH_STEPS[-1]["t_end"] is None:
                                    GROUND_TRUTH_STEPS[-1]["t_end"] = now
                            # evaluation done
                            cv2.putText(
                                frame,
                                "EVAL DONE - PRESS 'q' TO EXIT",
                                (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2
                            )
                        else:
                            if step_index != current_step_index:
                                # close previous step
                                if current_step_index >= 0 and GROUND_TRUTH_STEPS:
                                    if GROUND_TRUTH_STEPS[-1]["t_end"] is None:
                                        GROUND_TRUTH_STEPS[-1]["t_end"] = now

                                current_step_index = step_index
                                gt_gesture = SCRIPTED_GESTURES[step_index]
                                GROUND_TRUTH_STEPS.append({
                                    "index": step_index,
                                    "gesture": gt_gesture,
                                    "t_start": now,
                                    "t_end": None,
                                })
                                print(f"[GT] Step {step_index}: please do {gt_gesture}")

                            # Small msg to tell what to do now
                            gt_gesture = SCRIPTED_GESTURES[step_index]
                            text = f"DO NOW: {gt_gesture.upper()}"

                            cv2.rectangle(frame, (10, 10), (340, 80), (0, 0, 0), -1)
                            cv2.putText(
                                frame,
                                text,
                                (20, 55),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (0, 255, 255),
                                2
                            )

                    # Jump if both hands above shoulders
                    left_up = wl_sm < (shy_sm - HANDS_ABOVE_SHOULDERS_DELTA)
                    right_up = wr_sm < (shy_sm - HANDS_ABOVE_SHOULDERS_DELTA)
                    wrists_above = (left_up and right_up) if REQUIRE_BOTH_HANDS_FOR_JUMP else (left_up or right_up)

                    if wrists_above and (now - last_trigger_time["jump"] >= COOLDOWN_S):
                        #lag_ms = (time.time() - frame_ts) * 1000.0
                        lag_ms = (time.time() - frame_ts) * 1000.0 if "frame_ts" in locals() else 0.0
                        press("up")
                        last_trigger_time["jump"] = now
                        EVENT_LOG.append({"gesture": "jump", "time": now, "lag_ms": lag_ms})
                        print(f"[EVENT] jump | lag={lag_ms:.1f} ms")

                    # Duck if torso length shrinks vs standing reference
                    if standing_torso_ref is not None:
                        ratio = torso_sm / standing_torso_ref
                        if ratio < CROUCH_TORSO_RATIO and (now - last_trigger_time["duck"] >= COOLDOWN_S):
                            #lag_ms = (time.time() - frame_ts) * 1000.0
                            lag_ms = (time.time() - frame_ts) * 1000.0 if "frame_ts" in locals() else 0.0
                            press("down")
                            last_trigger_time["duck"] = now
                            EVENT_LOG.append({"gesture": "duck", "time": now, "lag_ms": lag_ms})
                            print(f"[EVENT] duck | lag={lag_ms:.1f} ms")

                    # Left / Right
                    if dx_sm <= -LEAN_THRESH and (now - last_trigger_time["left"] >= COOLDOWN_S):
                        #lag_ms = (time.time() - frame_ts) * 1000.0
                        lag_ms = (time.time() - frame_ts) * 1000.0 if "frame_ts" in locals() else 0.0
                        press("left")
                        last_trigger_time["left"] = now
                        EVENT_LOG.append({"gesture": "left", "time": now, "lag_ms": lag_ms})
                        print(f"[EVENT] left | lag={lag_ms:.1f} ms")

                    elif dx_sm >= LEAN_THRESH and (now - last_trigger_time["right"] >= COOLDOWN_S):
                        #lag_ms = (time.time() - frame_ts) * 1000.0
                        lag_ms = (time.time() - frame_ts) * 1000.0 if "frame_ts" in locals() else 0.0
                        press("right")
                        last_trigger_time["right"] = now
                        EVENT_LOG.append({"gesture": "right", "time": now, "lag_ms": lag_ms})
                        print(f"[EVENT] right | lag={lag_ms:.1f} ms")

                    if SHOW_DEBUG:
                        mp_drawing.draw_landmarks(
                            frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
                        )

                        sc = (int(sh_cx * w), int(sh_cy * h))
                        hc = (int(hip_cx * w), int(hip_cy * h))
                        cv2.circle(frame, sc, 6, (0, 255, 255), -1)
                        cv2.circle(frame, hc, 6, (255, 255, 0), -1)
                        cv2.line(frame, sc, hc, (200, 200, 50), 2)

                        # Shoulder reference line for jump
                        yline = int(shy_sm * h)
                        cv2.line(frame, (0, yline), (w, yline), (0, 120, 255), 1)

                        def put(y, txt):
                            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 50), 2, cv2.LINE_AA)

                        put(24,  f"Lean dx: {dx_sm:+.3f}  (thr {LEAN_THRESH})")
                        if standing_torso_ref is None:
                            put(48,  "Calibrating standing... hold upright")
                            put(72,  f"Torso: {torso_sm:.3f}")
                        else:
                            ratio = torso_sm / standing_torso_ref
                            put(48,  f"Torso ratio: {ratio:.3f} (duck<{CROUCH_TORSO_RATIO})")
                        put(72 if standing_torso_ref else 96, f"Wrists above: {wrists_above}")

            cv2.imshow("Temple Run Gestures (q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        if EVENT_LOG:
            with open("gesture_events.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["gesture", "event_time", "lag_ms"])
                for e in EVENT_LOG:
                    writer.writerow([e["gesture"], e["time"], e["lag_ms"]])
            print("Saved gesture_events.csv")


        if EVAL_MODE and GROUND_TRUTH_STEPS:
            if GROUND_TRUTH_STEPS[-1]["t_end"] is None:
                GROUND_TRUTH_STEPS[-1]["t_end"] = time.time()

        # evaluating accuracy
        if EVAL_MODE:
            from collections import Counter, defaultdict

            if not GROUND_TRUTH_STEPS:
                print("No ground truth steps recorded.")
            else:
                print("\nEVALUATION RESULTS : ")

                correct = 0
                total = 0
                confusion = defaultdict(lambda: Counter())

                events = EVENT_LOG

                for step in GROUND_TRUTH_STEPS:
                    gt = step["gesture"]
                    t0 = step["t_start"]
                    t1 = step["t_end"] or (t0 + STEP_DURATION)

                    step_events = [e for e in events if t0 <= e["time"] < t1]

                    if not step_events:
                        pred = "none"
                    else:
                        pred = step_events[0]["gesture"]

                    confusion[gt][pred] += 1
                    total += 1
                    if pred == gt:
                        correct += 1

                acc = correct / total if total > 0 else 0.0
                print(f"Total scripted gestures: {total}")
                print(f"Correctly detected:     {correct}")
                print(f"Accuracy:               {acc * 100:.1f}%\n")

                print("Confusion matrix (gt -> predicted counts):")
                for gt, row in confusion.items():
                    print(f"{gt:>5} -> {dict(row)}")

                with open("eval_ground_truth.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["index", "gesture", "t_start", "t_end"])
                    for s in GROUND_TRUTH_STEPS:
                        w.writerow([s["index"], s["gesture"], s["t_start"], s["t_end"]])

                with open("eval_events.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["gesture", "time", "lag_ms"])
                    for e in EVENT_LOG:
                        w.writerow([e["gesture"], e["time"], e["lag_ms"]])

                print("Saved eval_ground_truth.csv and eval_events.csv")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
