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

    ori_bb = ((pred_bb - np.array([left, top])) / ratio).astype(np.int)


    cv2.rectangle(ori_img, pt1=tuple(ori_bb[0]), pt2=tuple(ori_bb[1]), color=(255, 255, 255), thickness=10)

    plt.imshow(ori_img)
    plt.show()
if (os._exists(file_path)):
    os.mkdir(file_path)


for file_name in os.listdir(file_path):
    print(file_name)
    img = cv2.imread(file_path + '/' + file_name)
    ori_img = img.copy()
    img, ratio, top, left = resize_img(img)


    inputs = (img.astype('float32') / 255).reshape((1, img_size, img_size, 3))
    pred_bb = bbs_model.predict(inputs)[0].reshape((-1, 2))

    ori_bb = ((pred_bb - np.array([left, top])) / ratio).astype(np.int)


    cv2.rectangle(ori_img, pt1=tuple(ori_bb[0]), pt2=tuple(ori_bb[1]), color=(255, 255, 255), thickness=10)

    plt.imshow(ori_img)
    plt.show()