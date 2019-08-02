import dlib, cv2, os
from imutils import face_utils
import numpy as np
import matplotlib.pyplot as plt
import os
from functions import resize_img

predictor = dlib.shape_predictor('landmarkDetector.dat')
detector = dlib.cnn_face_detection_model_v1('dogHeadDetector.dat')

imgs_path = 'samples_dog'
imgs = []

fileList = list(os.listdir(imgs_path))
fileList.sort()

for img_path in fileList:
    if(img_path == '.DS_Store'):
        continue
    print(img_path)
    img = cv2.imread(imgs_path+ '/' + img_path)
    img, _a, _b, _c = resize_img(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    imgs.append(img)


# dlib 파일을 이용해 detection rec
detss = []
size = len(imgs)
img_results = []
del_list = []
first = True
print('실행')
for i, img in enumerate(imgs):
    print(str(i) + ' / ' + str(size))
    print(img.shape)
    dets = detector(img, upsample_num_times=1)
    if(len(dets) >= 2):
        print(i)
        del_list.append(i)
        continue
    detss.append(dets)

# 삭제된 img 파일 제
for i, del_count in enumerate(del_list):
    print(del_count - i)

    del imgs[del_count - i]


# 이미지, box, landmkarks 추출
res_shape = []
res = []
res_bbs = []
for i, img in enumerate(imgs):
    for j, d in enumerate(detss[i]):
        shape = predictor(img, d.rect)
        shape = face_utils.shape_to_np(shape)
        res.append(img)
        res_shape.append(shape)
        res_bbs.append(d.rect)

# 파일 저장
if not os._exists('./data'):
    os.mkdir('data')
for i, img in enumerate(res):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./data/' + str(i).zfill(4) + '.jpg', img)
    np.save('./data/' + str(i).zfill(4) + '.dog', res_shape[i])
    bbs = np.array((res_bbs[i].left(),res_bbs[i].top(),res_bbs[i].right(),res_bbs[i].bottom()))
    np.save('./data/' + str(i).zfill(4) + '.bbs', bbs)