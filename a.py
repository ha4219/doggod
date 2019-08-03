import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from functions import resize_img


path = 'data'


file_list = os.listdir(path)
file_list.sort()

X_train = []
y_train = []

for file_path in file_list:
    file_name, ext = os.path.splitext(file_path)
    print(file_name, ext)
    if (ext == '.npy' and 'dog' in file_name):
        y = np.load(path + '/' + file_path)
        print(y)
        y_train.append(y)

    elif (ext == '.jpg'):
        x = cv2.imread(path + '/' + file_path)
        X_train.append(x)

for i, img in enumerate(X_train):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ori_img = img.copy()

    for i, l in enumerate(y_train[i]):
        cv2.putText(ori_img, str(i), tuple(l), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(ori_img, center=tuple(l), radius=1, color=(255, 255, 255), thickness=3)

    plt.imshow(ori_img)
    plt.show()