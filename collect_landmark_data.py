import cv2
import mediapipe as mp
import csv
import os

# Create folders if they don't exist
DATA_DIR = 'landmark_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

gesture_name = input("Enter gesture label (e.g., mute, unmute, thumbs_up): ")
csv_file = os.path.join(DATA_DIR, f"{gesture_name}.csv")

# Open CSV file to append
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    cap = cv2.VideoCapture(0)
    print("Collecting data... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                landmarks = []
                for lm in handLms.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])  # 21 landmarks × 3 = 63 features
                writer.writerow(landmarks + [gesture_name])
                mp_draw.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Collecting Landmarks", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
