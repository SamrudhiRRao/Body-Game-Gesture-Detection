import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # Add this to your imports at the top



# --- Step 1: Load and Prepare Data ---

# Define the path to your new gesture data.
DATA_PATH = "knn_data" 

# Define the 8 specific gestures you want to train the model on.
# IMPORTANT: You must create an 'idle' folder in the DATA_PATH directory for this to work.
gestures = ['attack', 'crouch', 'jump', 'kick', 'move_left', 'move_right', 'quick_punch', 'idle']

X = [] # To store keypoints (features)
y = [] # To store gesture names (labels)

print(f"Loading data for the following gestures: {gestures}")

# Loop through each defined gesture
for gesture in gestures:
    gesture_path = os.path.join(DATA_PATH, gesture)
    if not os.path.isdir(gesture_path):
        print(f"Warning: Directory not found for gesture '{gesture}'. Skipping.")
        continue

    # Loop through each data file in the gesture folder
    for file_name in os.listdir(gesture_path):
        # Ensure we are only reading .npy files
        if not file_name.endswith(".npy"):
            continue
            
        file_path = os.path.join(gesture_path, file_name)
        
        try:
            # Load the keypoints from the .npy file.
            keypoints_data = np.load(file_path)
            
            # Flatten the keypoints array into a single list (1D feature vector)
            keypoints = keypoints_data.flatten()
            
            X.append(keypoints)
            y.append(gesture)
        except Exception as e:
            print(f"Could not read file {file_name}: {e}")

# Check if any data was actually loaded
if not X:
    print("Error: No data was loaded. Please check the DATA_PATH and ensure the gesture folders exist and contain .npy files.")
    exit()

print(f"Data loaded for {len(np.unique(y))} gestures. Total samples: {len(X)}")

# It's possible that not all feature vectors have the same length.
# KNN requires all feature vectors to be of the same dimension.
# Here's a simple padding strategy: find the max length and pad shorter vectors with 0.

max_len = max(len(features) for features in X)
X_padded = []
for features in X:
    padding = np.zeros(max_len - len(features))
    X_padded.append(np.concatenate((features, padding)))

X = np.array(X_padded)
y = np.array(y)

# --- Step 2: Train-Test Split ---

print("Splitting data into training and testing sets...")
# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# --- Step 3: Train the KNN Model ---

# Choose the number of neighbors (k). Start with an odd number.
K = 5 
model = KNeighborsClassifier(n_neighbors=K)

print(f"Training KNN model with k={K}...")
model.fit(X_train, y_train)
print("Model training complete.")

# --- Step 4: Evaluate the Model ---

print("Evaluating model...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Step 5: Save the Model ---
model_filename = "knn_gesture_model.pkl"
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")

print("KNN Gesture Recognition training complete.")

# You can now experiment with different values of 'k' or try different data preprocessing
# techniques to improve the accuracy.