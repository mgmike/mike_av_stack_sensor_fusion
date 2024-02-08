import sys
print(sys.version)

import cv2
print(cv2.__file__)
print(cv2.__version__)

import torch

from ultralytics import YOLO

# model = YOLO('data/models/yolov8m.pt')

# results = model('data/detect/mclaren-f1-various-supercars-1200-800.jpg', save=True)

# print(results[0].boxes)

# img = cv2.imread('data/detect/08869.jpg')
# img = cv2.resize(img, (640, 640))