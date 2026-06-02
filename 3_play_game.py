import cv2
import joblib
import pyautogui
import warnings
from collections import deque
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing_utils

# 1. Performance Settings
warnings.filterwarnings("ignore", category=UserWarning)
pyautogui.PAUSE = 0 # Removes the default delay in pyautogui
pyautogui.FAILSAFE = True 

# 2. Load the Brain
model = joblib.load('gesture_rf_model.pkl')

# 3. Fast MediaPipe Config (Complexity 0 is key for speed)
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.6, # Lowered slightly to be more responsive
    model_complexity=0 
)
# Try index 1 or 0 with cv2.CAP_DSHOW
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# Set camera resolution lower for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 4. Low-Latency Logic
scroll_buffer = deque(maxlen=2) # Only 2 frames needed to trigger!
last_pressed = None

print("TURBO MODE ACTIVE. Flick your hand and return to center to reset.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    
    current_gesture = "Neutral"

    if res.multi_hand_landmarks:
        for hand_landmarks in res.multi_hand_landmarks:
            # Fast Feature Extraction
            row = []
            wrist_x, wrist_y = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y
            for lm in hand_landmarks.landmark:
                row.extend([lm.x - wrist_x, lm.y - wrist_y])
            
            # Predict directly from list (Faster than DataFrame)
            prediction = model.predict([row])[0]
            scroll_buffer.append(prediction)

            if len(set(scroll_buffer)) == 1:
                action = scroll_buffer[0]
                current_gesture = action
                
                if action != "Neutral" and action != last_pressed:
                    pyautogui.press(action.lower())
                    last_pressed = action 
                elif action == "Neutral":
                    last_pressed = None

            # Optional: Comment these lines out to save even more speed
            drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Minimalist UI to save processing power
    cv2.putText(frame, f"ACT: {current_gesture}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Fast Controller', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()