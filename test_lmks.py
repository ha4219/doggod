from keras.models import Model, load_model
import os
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np


from functions import resize_img, overlay_transparent

img_size = 224
model_type = 'dog'
file_path = 'test_img'

# 이곳에 bbs 모델과 lmks 모델을 넣어주세요
bbs_model = load_model('models_' + model_type + '/bbs_1.h5')
lmks_model = load_model('models_' + model_type + '/lmks_1.h5')

if (os._exists(file_path)):
    os.mkdir(file_path)


for file_name in os.listdir(file_path):
    print(file_name)
    img = cv2.imread(file_path + '/' + file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ori_img = img.copy()
    img, ratio, top, left = resize_img(img)

    inputs = (img.astype('float32') / 255).reshape((1, img_size, img_size, 3))
    pred_bb = bbs_model.predict(inputs)[0].reshape((-1, 2))

    # compute bounding box of original image
    ori_bb = ((pred_bb - np.array([left, top])) / ratio).astype(np.int)

    # compute lazy bounding box for detecting landmarks
    center = np.mean(ori_bb, axis=0)
    face_size = max(np.abs(ori_bb[1] - ori_bb[0]))
    new_bb = np.array([
        center - face_size * 0.6,
        center + face_size * 0.6
    ]).astype(np.int)
    new_bb = np.clip(new_bb, 0, 99999)

    # predict landmarks
    face_img = ori_img[new_bb[0][1]:new_bb[1][1], new_bb[0][0]:new_bb[1][0]]
    face_img, face_ratio, face_top, face_left = resize_img(face_img)

    face_inputs = (face_img.astype('float32') / 255).reshape((1, img_size, img_size, 3))

    pred_lmks = lmks_model.predict(face_inputs)[0].reshape((-1, 2))

    # compute landmark of original image
    new_lmks = ((pred_lmks - np.array([face_left, face_top])) / face_ratio).astype(np.int)
    ori_lmks = new_lmks + new_bb[0]

    # visualize
    cv2.rectangle(ori_img, pt1=tuple(ori_bb[0]), pt2=tuple(ori_bb[1]), color=(0, 0, 255), thickness=2)

    for i, l in enumerate(ori_lmks):
        cv2.putText(ori_img, str(i), tuple(l), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(ori_img, center=tuple(l), radius=1, color=(255, 255, 255), thickness=2)

    plt.imshow(ori_img)
    plt.show()