import cv2
import os
import numpy as np
import mediapipe as mp
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Video Capture
cap = cv2.VideoCapture(0)
gestures = ['increase_volume', 'decrease_volume', 'mute', 'unmute']

# Create directories
for gesture in gestures:
    os.makedirs(f'dataset/{gesture}', exist_ok=True)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

current_gesture = gestures[0]
count = 0

print(f"Collecting for: {current_gesture}")
print("Press 'n' to switch gesture, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size > 0:
                gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (128, 128))
                expanded = np.expand_dims(resized, axis=(0, -1))
                augmented = datagen.flow(expanded, batch_size=1)

                for i in range(2):
                    aug_img = next(augmented)[0].astype('uint8')
                    path = f'dataset/{current_gesture}/img_{count}_{i}.jpg'
                    cv2.imwrite(path, aug_img)
                    print(f"Saved: {path}")

                count += 1

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1)
    if key == ord('n'):
        count = 0
        current_index = (gestures.index(current_gesture) + 1) % len(gestures)
        current_gesture = gestures[current_index]
        print(f"Switched to: {current_gesture}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
