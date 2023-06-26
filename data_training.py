import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

from keras.layers import Input, Dense
from keras.models import Model 

is_initialize = False
size = -1

label = []
dct = {}
c = 0

for i in os.listdir():
    if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):
        if not is_initialize:
            is_initialize = True
            data = np.load(i)
            size = data.shape[0]
            y = np.array([i.split('.')[0]]*size).reshape(-1,1)
        else:
            data = np.concatenate((data, np.load(i)))
            y = np.concatenate((y, np.array([i.split('.')[0]]*size).reshape(-1,1)))

        label.append(i.split('.')[0])
        dct[i.split('.')[0]] = c
        c = c+1


for i in range(y.shape[0]):
    y[i, 0] = dct[y[i, 0]]
y = np.array(y,dtype="int32")



y = to_categorical(y)

newdata = data.copy()
newy = y.copy()
counter = 0

cnt = np.arange(data.shape[0])
np.random.shuffle(cnt)

for i in cnt:
    newdata[counter] = data[i]
    newy[counter] = y[i]
    counter = counter+1


ip = Input(shape=(data.shape[1]))

m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)

op = Dense(y.shape[1], activation="softmax")(m)

model = Model(inputs=ip, outputs=op)

model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

model.fit(data, y, epochs=50)


model.save("model.h5")
np.save("labels.npy", np.array(label))