import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyautogui

# Load trained model
model = joblib.load('gesture_landmark_model.pkl')

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert color
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            if len(landmark_list) == 63:
                prediction = model.predict([landmark_list])[0]
                
                # Display prediction on screen
                cv2.putText(frame, f'Gesture: {prediction}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 🔊 Perform system volume actions
                if prediction == "mute":
                    pyautogui.press('volumemute')
                elif prediction == "increase_volume":
                    pyautogui.press('volumeup')
                elif prediction == "decrease_volume":
                    pyautogui.press('volumedown')

    # Show output
    cv2.imshow('Hand Gesture Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
