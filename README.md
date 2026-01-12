## Gesture Control Karate System

This project uses computer vision and machine learning to control a karate game using real-time body gestures. It captures your pose via a webcam, classifies the move (e.g., Punch, Kick, Crouch), and simulates the corresponding keyboard press.

📋 Prerequisites

Ensure you have Python installed along with the required libraries:
```
pip install opencv-python mediapipe numpy joblib pandas scikit-learn pydirectinput
```

🚀 Usage Instructions

Step 1: Train the Model

Before running any gesture recognition, you must generate the model file.

Script: optimized_trainer.py

Input: Requires karate_optimized_data.csv (your dataset) in the same directory.

Action: Run this script to train the K-Nearest Neighbors (KNN) classifier.

Output: Generates optimized_karate_model.pkl.
```
python optimized_trainer.py
```

Step 2: Run the Game Controller (Main Mode)

This is the primary script for playing the game.

Script: gameplay_with_KNN.py

Function: Loads the trained model, detects your gestures, and simulates actual key presses to control your game character.

### Controls:
### Punch: L (Low), I (High)
### Kick:  K
### Move:  W (Jump), S (Crouch), A (Left), D (Right)
### Combo: U
```
python gameplay_with_KNN.py
```

Note: Switch focus to your game window immediately after running this script.

Step 3: View Performance Metrics (Additional Mode)

This script is an additional tool for testing and visualization. It relies on the same logic and model as the gameplay script but does not press keys.

Script: new_live_metrics.py

Function: Displays a futuristic HUD showing system performance including:

FPS (Frames Per Second): Monitors video smoothness.

Latency: Measures processing delay in milliseconds.

Confidence: Visual bar showing how sure the model is of your pose.

Use Case: Run this to debug lag or accuracy issues without triggering game inputs.
```
python new_live_metrics.py
```

⚠️ Troubleshooting

FileNotFoundError: If you see Error: Could not find optimized_karate_model.pkl, ensure you have run optimized_trainer.py first.

Lighting: Ensure your room is well-lit so MediaPipe can accurately detect your body landmarks.