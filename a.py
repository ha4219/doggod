import cv2
from functions import resize_img
img = cv2.imread('./samples_dog/dog00001.jpg')
img, a, b, c = resize_img(img)

cv2.imshow('s',img)