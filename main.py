import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

model = load_model('gesture_model.h5')
gestures = ['increase_volume', 'decrease_volume', 'mute', 'unmute']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in landmarks.landmark]) * h)

            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size > 0:
                gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (128, 128))
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, 128, 128, 1))
                prediction = model.predict(reshaped)
                label = gestures[np.argmax(prediction)]
                confidence = np.max(prediction)

                cv2.putText(frame, f'{label} ({confidence:.2f})', (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Gesture Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
