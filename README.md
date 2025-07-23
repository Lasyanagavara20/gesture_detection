# Gesture Detection with Hand Landmarks

This project uses MediaPipe and a trained Random Forest model to recognize hand gestures from webcam input. It supports gestures like:

- ✋ Mute
- 🔊 Increase Volume
- 🔉 Decrease Volume

## 🛠 Tech Stack
- Python
- OpenCV
- MediaPipe
- scikit-learn
- pyautogui

## 🚀 Getting Started

1. Clone the repo:
git clone https://github.com/Lasyanagavara20/gesture_detection.git 

2. Install dependencies:
pip install -r requirements.txt

3. Run detection:
python detect_landmark_gesture.py

## 🤖 Training Data

You can collect and train new gestures using:
- `collect_landmark_data.py`
- `merge_landmark_data.py`
- `train_landmark_model.py`

---



