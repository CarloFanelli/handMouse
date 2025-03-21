import os
import random
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options
import pyautogui
import numpy as np

# Configurazione iniziale
os.add_dll_directory(r'C:\Users\CarloFanelli\AppData\Local\Programs\Python\Python312\Lib\site-packages\mediapipe\python')

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



# Variabili globali
gesture_name = ''
action_name = ''
score = 0

class HandPoint:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

indexFingerTip = HandPoint()
thumbFingerTip = HandPoint()
indexFingerBase = HandPoint()
middleFingerTip = HandPoint()




# Aggiorna le informazioni sul gesto riconosciuto
def update_gesture_info(gesture_result):
    global gesture_name, score
    if gesture_result.gestures and len(gesture_result.gestures) > 0:
        for gesture in gesture_result.gestures:
            if gesture and len(gesture) > 0:
                gesture_name = gesture[0].category_name
                score = gesture[0].score



#verifica il pinch e disegna un punto nel centro del pinch
def checkPinch():
    distance = np.sqrt((indexFingerTip.x - thumbFingerTip.x)**2 + (indexFingerTip.y - thumbFingerTip.y)**2 + (indexFingerTip.z - thumbFingerTip.z)**2)
    threshold = 0.06  # Puoi regolare questo valore in base alla precisione desiderata
    is_pinching = distance < threshold

    # Calcola il centro della distanza
    center_x = 1- ((indexFingerTip.x + thumbFingerTip.x) / 2)
    center_y = (indexFingerTip.y + thumbFingerTip.y) / 2
    center_z = (indexFingerTip.z + thumbFingerTip.z) / 2
    handPoint = HandPoint(center_x, center_y, center_z)

    return is_pinching, distance,handPoint


#fai click (punta del pollice tocca punta del medio)
def click():
    distance = np.sqrt((middleFingerTip.x - thumbFingerTip.x)**2 + (middleFingerTip.y - thumbFingerTip.y)**2 + (middleFingerTip.z - thumbFingerTip.z)**2)
    threshold = 0.045  # Puoi regolare questo valore in base alla precisione desiderata
    clicking = distance < threshold

    # Calcola il centro della distanza
    center_x = 1- ((middleFingerTip.x + thumbFingerTip.x) / 2)
    center_y = (middleFingerTip.y + thumbFingerTip.y) / 2
    center_z = (middleFingerTip.z + thumbFingerTip.z) / 2
    handPoint = HandPoint(center_x, center_y, center_z)

    return clicking, distance,handPoint


# Disegna un cerchio al centro della distanza tra indice e pollice
def draw_circle_at_center(image, center, distance):
    height, width, _ = image.shape
    center_point = (int((1 - center.x) * width), int(center.y * height))
    radius = int(distance * 600) 
    cv2.circle(image, center_point, radius, (0, 0, 255), 2)


# Aggiorna le posizioni dei punti della mano
def update_hand_points(gesture_result):
    global indexFingerTip, thumbFingerTip,indexFingerBase,middleFingerTip
    if gesture_result.hand_landmarks:
        for hand_landmarks in gesture_result.hand_landmarks:
            if hand_landmarks and len(hand_landmarks) > 8:
                indexFingerTip = HandPoint(
                    x=1 - hand_landmarks[8].x,
                    y=hand_landmarks[8].y,
                    z=hand_landmarks[8].z
                )
                thumbFingerTip = HandPoint(
                    x=1 - hand_landmarks[4].x,
                    y=hand_landmarks[4].y,
                    z=hand_landmarks[4].z
                )
                indexFingerBase = HandPoint(
                    x=1 - hand_landmarks[5].x,
                    y=hand_landmarks[5].y,
                    z=hand_landmarks[5].z
                )
                middleFingerTip = HandPoint(
                    x=1 - hand_landmarks[12].x,
                    y=hand_landmarks[12].y,
                    z=hand_landmarks[12].z
                )




# Disegna i punti e le connessioni della mano
def draw_landmarks(image, hand_landmarks):
    height, width, _ = image.shape
    landmarks_list = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand_landmarks]

    for connection in mp_hands.HAND_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(landmarks_list) and end_idx < len(landmarks_list):
            start_point = (int(landmarks_list[start_idx]['x'] * width), int(landmarks_list[start_idx]['y'] * height))
            end_point = (int(landmarks_list[end_idx]['x'] * width), int(landmarks_list[end_idx]['y'] * height))
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)

    for landmark in landmarks_list:
        pos = (int(landmark['x'] * width), int(landmark['y'] * height))
        cv2.circle(image, pos, 5, (0, 0, 255), -1)




# Disegna una linea tra due punti
def draw_line_between_points(image, point1, point2):
    height, width, _ = image.shape
    start_point = (int((1- point1.x) * width), int(point1.y * height))
    end_point = (int((1-point2.x) * width), int(point2.y * height))
    cv2.line(image, start_point, end_point,(255, 255, 255), 3) #edit here to pass the color to the function




# Elabora un frame dell'immagine per il riconoscimento dei gesti
def process_frame(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    gesture_result = gesture_recognizer.recognize(mp_image)
    update_gesture_info(gesture_result)
    update_hand_points(gesture_result)
    return gesture_result




# Visualizza le informazioni sull'immagine
def display_info(image):
    text = f'screen  x: {screen_width} | y: {screen_height}'
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    text2 = f'indexFingerTip  x: {indexFingerTip.x:.2f} | y: {indexFingerTip.y:.2f} | z: {indexFingerTip.z:.2f}' if indexFingerTip.x or indexFingerTip.y else 'indexFingerTip not found'
    cv2.putText(image, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
    
    text3 = f'thumbFingerTip  x: {thumbFingerTip.x:.2f} | y: {thumbFingerTip.y:.2f} | z: {thumbFingerTip.z:.2f}' if thumbFingerTip.x or thumbFingerTip.y else 'thumbFingerTip not found'
    cv2.putText(image, text3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
    
    cv2.putText(image, f'Gesto: {gesture_name} ({score:.2f})', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    isPinching,distance,handPoint = checkPinch()



    if not isPinching:
        draw_circle_at_center(image_display,handPoint,(distance))

    #debug pinch
    # cv2.putText(image, f'pinch: {isPinching} - {distance}', (width-20, height-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    #debug click
    clicking, distance2,handPoint = click()

    cv2.putText(image, f'click: {clicking} - {distance2}', (width-20, height-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


#verifica se il gesto è pointing_up
def isPointing():
    return gesture_name == 'Pointing_Up'




#code
while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Ignoring empty frame")
        continue

    gesture_result = process_frame(img)
    image_display = img.copy()
    height, width, _ = image_display.shape


    if gesture_result.hand_landmarks:
        for hand_landmarks in gesture_result.hand_landmarks:
            draw_landmarks(image_display, hand_landmarks)
        # draw_line_between_points(image_display, thumbFingerTip, indexFingerTip)
        # draw_line_between_points(image_display, thumbFingerTip, indexFingerBase)
        draw_line_between_points(image_display, thumbFingerTip, middleFingerTip)

    image_display = cv2.resize(image_display, (screen_width, screen_height))
    image_display = cv2.flip(image_display, 1)
    display_info(image_display)
    cv2.imshow('Hand Tracking', image_display)

    if indexFingerTip.x and indexFingerTip.y and indexFingerTip.z and isPointing():
        # pyautogui.moveTo((indexFingerTip.x) * screen_width, (indexFingerTip.y) * screen_height)

        clicking, distance,handPoint = click()
        if clicking:
            print('click',distance)
            # pyautogui.click()
        else:
            pass
    else:
        pass

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()