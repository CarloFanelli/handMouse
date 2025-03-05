import os
os.add_dll_directory(r'C:\Users\CarloFanelli\AppData\Local\Programs\Python\Python312\Lib\site-packages\mediapipe\python')
import cv2
import mediapipe as mp

#riconoscere le mani
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

#webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success,img = cap.read()
    if not success:
        print("ignoro frame vuoto")
        continue

    #converto RGB
    image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = hands.process(image)

    #torno in BGR
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

    #disegno sulle mani
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS)

    #MOSTRO L'immagine
    cv2.imshow('hands',image)

    #con esc chiudo
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
print('closed')
cv2.destroyAllWindows()