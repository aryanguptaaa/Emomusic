import mediapipe as mp 
import numpy as np
import cv2

cp = cv2.VideoCapture(0)
nm = input("Enter name of the emotion : ")

holistic = mp.solutions.holistic
hnds = mp.solutions.hands
hol = holistic.Holistic()
drawings = mp.solutions.drawing_utils

data = []
data_size = 0

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


        data.append(lst)
        data_size = data_size+1

    drawings.draw_landmarks(frame, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawings.draw_landmarks(frame, res.left_hand_landmarks, holistic.HAND_CONNECTIONS)
    drawings.draw_landmarks(frame, res.right_hand_landmarks, holistic.HAND_CONNECTIONS)

    cv2.putText(frame, str(data_size), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

    cv2.imshow("window", frame)

    if cv2.waitKey(1) == 27 or data_size>99:
        cv2.destroyAllWindows()
        cp.release()
        break


np.save(f"{nm}.npy", np.array(data))
print(np.array(data).shape)