import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load CSV
df = pd.read_csv("landmark_data.csv", header=None)

# Last column is the label
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Model trained successfully.")

# ✅ Save the model
joblib.dump(model, 'gesture_landmark_model.pkl')
print("Model saved to gesture_landmark_model.pkl")
