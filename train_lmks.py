import keras, datetime
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.applications import mobilenetv2
import numpy as np
import os
import cv2


'''
    주석은 bbs 부분에 달아 놨습니다. bbs 코드와 매우 유사해요 
'''
img_size = 224

output_size = 12

path = './data'
file_list = os.listdir(path)
file_list.sort()

X_train = []
y_train = []

for file_path in file_list:
    file_name, ext = os.path.splitext(file_path)
    print(file_name, ext)
    if (ext == '.npy' and 'dog' in file_name):
        y = np.load(path + '/' + file_path)
        y = y.flatten()
        print(y)
        y_train.append(y)

    elif(ext == '.jpg'):
        x = cv2.imread(path + '/' + file_path)
        X_train.append(x)

X_train = np.array(X_train)
y_train = np.array(y_train)
print(X_train.shape)
print(y_train.shape)

X_train = X_train / 255.

X_test = X_train[-10:]
X_train = X_train[:-10]
y_test = y_train[-10:]
y_train = y_train[:-10]

start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

inputs = Input(shape=(img_size, img_size, 3))
#, depth_multiplier=1
mobilenetv2_model = mobilenetv2.MobileNetV2(input_shape=(img_size, img_size, 3), alpha=1.0, include_top=False, weights='imagenet', input_tensor=inputs, pooling='max')

net = Dense(128, activation='relu')(mobilenetv2_model.layers[-1].output)
net = Dense(64, activation='relu')(net)
net = Dense(output_size, activation='linear')(net)

model = Model(inputs=inputs, outputs=net)

model.summary()

# training
model.compile(optimizer=keras.optimizers.Adam(), loss='mse')

model.fit(X_train, y_train, epochs=50, batch_size=32, shuffle=True,
  validation_data=(X_test, y_test), verbose=1,
  callbacks=[
    TensorBoard(log_dir='logs/%s' % (start_time)),
    ModelCheckpoint('./models_dog/%s.h5' % ('lmks_1'), monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto')
  ]
)
