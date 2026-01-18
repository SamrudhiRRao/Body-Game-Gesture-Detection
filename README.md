## Gesture Control Karate System

This project uses computer vision and machine learning to control a karate game using real-time body gestures. It captures your pose via a webcam, classifies the move (e.g., Punch, Kick, Crouch), and simulates the corresponding keyboard press.

📋 Prerequisites

Ensure you have Python installed along with the required libraries:
```
pip install opencv-python mediapipe pandas numpy scikit-learn joblib pydirectinput
```

## 🚀 Usage Instructions

Follow these three steps to set up and run the controller.

### Step 1: Data Collection
First, you need to record your own body gestures to create the dataset.

1. Run the collection script:

```
python optimized_collection.py
```
2. Select a Gesture: Press `N` to toggle between gestures (e.g., Punch, Kick, Jump).

3. Record: Press `SPACE` to start.
    - The system acts as a 10-second timer.
    - After a 3-second countdown, perform the gesture continuously for 10 seconds.

4. Repeat this for all 8 gestures to populate `karate_optimized_data.csv`




### Step 2: Train the Model

Once the data is collected, train the machine learning model.

1. Run the training script:


```
python optimized_trainer.py
```
2. This script processes the CSV file, trains a KNN classifier, and exports the weights to `optimized_karate_model.pkl`.

3. Note: Ensure the `.pkl` file is saved in the same directory as the gameplay script.


### Step 3: Run the Controller

Now you are ready to play.

1. Open your target game in a web browser (e.g., Karate Fighter).

2. Run the gameplay script:


```
python gameplay_with_KNN.py
```

3. Focus the Window: Click on the game window once to ensure it receives keyboard inputs.

4. Stand back and control the character with your movements!

## 🎮 Controls

The system maps the following physical gestures to keyboard keys:

| Gesture | Key Mapped | Action |
| ---------- | ---------- | ---------- |
| Extend both arms to the sides|W|Character Jumps|
| Crouch|S|Character Crouches|
| Extend Arm to Left|A|Move Left|
| Extend Arm to Right|D|Move Right|
| Make a Fist with Left or Right|L|Low Punch|
| Make a Fist with both Left and Right|I|High Punch|
| Lift and Extend Right Leg Forward|K|Strong Kick Attack|
| Lift Left Leg just above ground| J |High Kick Attack|
| Wakanda Forever| U |Combo Hit|
| Stand straight with Arms Extended down|-|No Move, Stays Idle|

### 🎬 Demonstration Video
▶️ [Watch Demo Video]([https://drive.google.com/file/d/1lTuqTSe5ODM5KuVjWyF_74F4nHisOm5g/view?usp=sharing])


## ⚠️ Troubleshooting

- Camera not opening? Ensure no other application (Zoom, Teams) is using the webcam.

- Model not found? Verify that optimized_karate_model.pkl exists in the root folder before running the gameplay script.

- Low Accuracy? Try re-running Step 1 and recording more diverse samples for the problematic gesture.
