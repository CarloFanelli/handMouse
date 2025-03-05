import os
os.add_dll_directory(r'C:\Users\CarloFanelli\AppData\Local\Programs\Python\Python312\Lib\site-packages\mediapipe\python')
import cv2
import mediapipe as mp
import pyautogui


#riconoscere le mani
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

#webcam
cap = cv2.VideoCapture(0)

cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Hand Tracking', cv2.WINDOW_AUTOSIZE, cv2.WINDOW_NORMAL)
# screen_width = cv2.getWindowImageRect('Hand Tracking')[2]
# screen_heigth = cv2.getWindowImageRect('Hand Tracking')[3]
screen_width = pyautogui.size().width
screen_heigth = pyautogui.size().height
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

    indexFinger = {'x':0, 'y':0}

    #disegno sulle mani
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS)
            indexFinger = {'x':1-hand_landmarks.landmark[8].x, 'y':hand_landmarks.landmark[8].y}

    #full screen effetto specchio
    image = cv2.resize(image,(screen_width,screen_heigth))
    image = cv2.flip(image,1)

    #scrivo sull'immagine
    text = 'screen  x: '+str(screen_width)+' | y: '+str(screen_heigth) 
    cv2.putText(image,text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),
                2,cv2.LINE_AA)
    
    text2 = 'indexFinger  x: '+str(indexFinger['x'])+' | y: '+str(indexFinger['y']) if indexFinger else 'indexFinger not found'
    cv2.putText(image,text2,(10,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),
                2,cv2.LINE_4)

    #MOSTRO L'immagine
    cv2.imshow('Hand Tracking',image)
    if indexFinger['x'] and indexFinger['y']:
        pyautogui.moveTo(indexFinger['x']*screen_width,indexFinger['y']*screen_heigth)
    #con esc chiudo
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
print('closed')
cv2.destroyAllWindows()