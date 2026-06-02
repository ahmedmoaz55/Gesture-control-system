import cv2
import csv
import time
import os

# --- THE BULLETPROOF IMPORTS (BYPASSES THE WINDOWS BUG) ---
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing_utils
import mediapipe.python.solutions.drawing_styles as drawing_styles

# Initialize MediaPipe directly without using 'mp.solutions'
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Setup Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

gesture = input("Enter gesture to record (e.g., Up, Down, Left, Right, Neutral): ")
filename = 'hand_data.csv'

# Create headers if new file
if not os.path.exists(filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = []
        for i in range(21):
            headers.extend([f'x_{i}', f'y_{i}'])
        headers.append('label')
        writer.writerow(headers)

print(f"Get ready to perform '{gesture}' in 3 seconds...")
time.sleep(3)
print("Recording... Press 'q' to stop.")

count = 0
with open(filename, 'a', newline='') as f:
    writer = csv.writer(f)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                row = []
                # Wrist Normalization
                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y
                
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x - wrist_x, lm.y - wrist_y])
                
                row.append(gesture)
                writer.writerow(row)
                count += 1
                
                # Draw the hand using the bypass imports
                drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style()
                )

        cv2.putText(frame, f"Frames: {count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Recording Data', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Saved {count} frames of '{gesture}'.")