import keras, datetime
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.applications import mobilenetv2
import numpy as np
import os
import cv2

# img_size input 데이터의 이미지 크기를 의미
# 예를 들면 shape 를 찍으면 (224, 224, 3) 이런식에 데이터
img_size = 224

# 결과 최종 적으로 bbs는 박스 형태로 (left, top), (right, bottom) 이렇게 점을 2개를 보여줌으로
# output 값은 4이다.
output_size = 4

# file load
path = './data'
file_list = os.listdir(path)
file_list.sort()

X_train = []
y_train = []

for file_path in file_list:
    file_name, ext = os.path.splitext(file_path)
    print(file_name, ext)
    if (ext == '.npy' and 'bbs' in file_name):
        y = np.load(path + '/' + file_path)
        y = y.flatten()
        print(y)
        y_train.append(y)

    elif (ext == '.jpg'):
        x = cv2.imread(path + '/' + file_path)
        X_train.append(x)

# 파일을 numpy 배열로 바꿔줌
X_train = np.array(X_train)
y_train = np.array(y_train)
print(X_train.shape)
print(y_train.shape)

# 정규
X_train = X_train / 255.

#데이터가 별로 없어서 10개로 test data 정함 수정 가
X_test = X_train[-10:]
X_train = X_train[:-10]
y_test = y_train[-10:]
y_train = y_train[:-10]


# 시작 시간
start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

inputs = Input(shape=(img_size, img_size, 3))

# erorr -> depth_multiplier=1
mobilenetv2_model = mobilenetv2.MobileNetV2(input_shape=(img_size, img_size, 3), alpha=1.0, include_top=False, weights='imagenet', input_tensor=inputs, pooling='max')

net = Dense(128, activation='relu')(mobilenetv2_model.layers[-1].output)
net = Dense(64, activation='relu')(net)
net = Dense(output_size, activation='linear')(net)

model = Model(inputs=inputs, outputs=net)

model.summary()

# training
model.compile(optimizer=keras.optimizers.Adam(), loss='mse')

# keras 에서 모델을 자동 저장해줌 save_best_only 저부분
model.fit(X_train, y_train, epochs=50, batch_size=32, shuffle=True,
  validation_data=(X_test, y_test), verbose=1,
  callbacks=[
    TensorBoard(log_dir='logs/%s' % (start_time)),
    ModelCheckpoint('./models_dog/%s.h5' % ('bbs_1'), monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto')
  ]
)
