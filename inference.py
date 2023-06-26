import mediapipe as mp 
import numpy as np
import cv2
from keras.models import load_model

model = load_model("model.h5")
label = np.load("labels.npy")


holistic = mp.solutions.holistic
hnds = mp.solutions.hands
hol = holistic.Holistic()
drawings = mp.solutions.drawing_utils

cp = cv2.VideoCapture(0)


while True:
    lst = []

    _, frame = cp.read()
    frame = cv2.flip(frame, 1)
    res = hol.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)


        lst = np.array(lst).reshape(1,-1)

        pred = label[np.argmax(model.predict(lst))]

    drawings.draw_landmarks(frame, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawings.draw_landmarks(frame, res.left_hand_landmarks, holistic.HAND_CONNECTIONS)
    drawings.draw_landmarks(frame, res.right_hand_landmarks, holistic.HAND_CONNECTIONS)

    cv2.imshow("window", frame)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cp.release()
        break
