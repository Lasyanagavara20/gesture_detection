import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

DATA_DIR = 'dataset'
CATEGORIES = os.listdir(DATA_DIR)
data = []
labels = []

print("Loading data...")
for category in CATEGORIES:
    folder_path = os.path.join(DATA_DIR, category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 64))  # Resize for consistency
        data.append(img.flatten())
        labels.append(category)

print("Data loaded:", len(data), "samples")

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"Model accuracy: {acc*100:.2f}%")

# Save the model
joblib.dump(model, 'gesture_model.pkl')
print("Model saved as gesture_model.pkl")
