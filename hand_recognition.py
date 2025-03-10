import os
os.add_dll_directory(r'C:\Users\CarloFanelli\AppData\Local\Programs\Python\Python312\Lib\site-packages\mediapipe\python')
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options
import pyautogui
import numpy as np

model_path = './gesture_recognizer.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)

gesture_recognizer = vision.GestureRecognizer.create_from_options(options)

# Per disegnare i landmark
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Webcam
cap = cv2.VideoCapture(0)

cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Hand Tracking', cv2.WINDOW_AUTOSIZE, cv2.WINDOW_NORMAL)
screen_width = pyautogui.size().width
screen_height = pyautogui.size().height

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Ignoring empty frame")
        continue

    # Converti in RGB per MediaPipe
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Crea l'oggetto mp.Image correttamente
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    
    # Riconoscimento del gesto
    gesture_result = gesture_recognizer.recognize(mp_image)
    
    # Converti di nuovo in BGR per la visualizzazione
    image_display = img.copy()  # Usa l'immagine originale per la visualizzazione
    
    indexFinger = {'x': 0, 'y': 0}
    
    gesture_name = ''
    # Se ci sono gesti riconosciuti, visualizzali
    if gesture_result.gestures and len(gesture_result.gestures) > 0:
        for i, gesture in enumerate(gesture_result.gestures):
            if gesture and len(gesture) > 0:
                gesture_name = gesture[0].category_name
                score = gesture[0].score
                print(f'Gesto riconosciuto: {gesture_name} (score: {score:.2f})')
                
    # Se sono rilevati landmarks delle mani, visualizzali
    
    # if gesture_result.hand_landmarks:
        for idx, hand_landmarks in enumerate(gesture_result.hand_landmarks):
            # Calcola le coordinate normalizzate dell'indice per il controllo del mouse
            if hand_landmarks and len(hand_landmarks) > 8:  # Assicurati che ci sia l'indice (landmark 8)
                indexFinger = {
                    'x': 1 - hand_landmarks[8].x,  # Inverti l'asse x per effetto specchio
                    'y': hand_landmarks[8].y
                }
                
                # Conversione per il disegno dei landmark
                height, width, _ = image_display.shape
                hand_landmarks_proto = []
                
                # Creiamo una struttura compatibile con mp_drawing
                landmarks_list = []
                for landmark in hand_landmarks:
                    landmarks_list.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                
                # Disegna i landmarks manualmente
                connections = mp_hands.HAND_CONNECTIONS
                for connection in connections:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    
                    if start_idx < len(landmarks_list) and end_idx < len(landmarks_list):
                        start_point = (int(landmarks_list[start_idx]['x'] * width), 
                                      int(landmarks_list[start_idx]['y'] * height))
                        end_point = (int(landmarks_list[end_idx]['x'] * width), 
                                     int(landmarks_list[end_idx]['y'] * height))
                        
                        cv2.line(image_display, start_point, end_point, (0, 255, 0), 2)
                
                # Disegna i punti
                for idx, landmark in enumerate(landmarks_list):
                    pos = (int(landmark['x'] * width), int(landmark['y'] * height))
                    cv2.circle(image_display, pos, 5, (0, 0, 255), -1)
    
    # Effetto specchio a schermo intero
    image_display = cv2.resize(image_display, (screen_width, screen_height))
    image_display = cv2.flip(image_display, 1)  # Flip orizzontale

    # Scrivi sull'immagine
    text = f'screen  x: {screen_width} | y: {screen_height}'
    cv2.putText(image_display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    text2 = f'indexFinger  x: {indexFinger["x"]:.2f} | y: {indexFinger["y"]:.2f}' if indexFinger['x'] or indexFinger['y'] else 'indexFinger not found'
    cv2.putText(image_display, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)

    cv2.putText(image_display, f'Gesto: {gesture_name} ({score:.2f})',(10, 90 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # Mostra l'immagine
    cv2.imshow('Hand Tracking', image_display)
    
    # Muovi il mouse se l'indice è rilevato
    if indexFinger['x'] and indexFinger['y']:
        pyautogui.moveTo(indexFinger['x'] * screen_width, indexFinger['y'] * screen_height)
    
    # Chiudi con ESC
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
print('closed')
cv2.destroyAllWindows()