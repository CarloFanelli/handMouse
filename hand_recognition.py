import os
os.add_dll_directory(r'C:\Users\CarloFanelli\AppData\Local\Programs\Python\Python312\Lib\site-packages\mediapipe\python')
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options  # Importa base_options
import pyautogui

model_path = './gesture_recognizer.task'
base_options = base_options.BaseOptions(model_asset_path=model_path)

# Recognize hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Hand Tracking', cv2.WINDOW_AUTOSIZE, cv2.WINDOW_NORMAL)
screen_width = pyautogui.size().width
screen_heigth = pyautogui.size().height

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Ignoring empty frame")
        continue

    # Convert to RGB
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(image)

    # Convert back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    indexFinger = {'x': 0, 'y': 0}

    # Draw landmarks on hands
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            indexFinger = {'x': 1 - hand_landmarks.landmark[8].x, 'y': hand_landmarks.landmark[8].y}
            
    # Full screen mirror effect
    image = cv2.resize(image, (screen_width, screen_heigth))
    image = cv2.flip(image, 1)

    # Write on the image
    text = f'screen  x: {screen_width} | y: {screen_heigth}'
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    text2 = f'indexFinger  x: {indexFinger["x"]} | y: {indexFinger["y"]}' if indexFinger else 'indexFinger not found'
    cv2.putText(image, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)

    # Show the image
    cv2.imshow('Hand Tracking', image)
    if indexFinger['x'] and indexFinger['y']:
        pyautogui.moveTo(indexFinger['x'] * screen_width, indexFinger['y'] * screen_heigth)
    # Close with ESC
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
print('closed')
cv2.destroyAllWindows()