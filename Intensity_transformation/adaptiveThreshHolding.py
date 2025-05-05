import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../image/sudoku-original.jpg", 0)
ret,thr1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
thr2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,2)
thr3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2)

titles = [
    'Original Image',
    '1. Global Thresh Holding',
    '2. Adaptive  Thresh Holding MEAN_C',
    '3. Adaptive  Thresh Holding GAUSSIAN_C',

]
images = [img, thr1,thr2,thr3]

plt.figure(figsize=(12, 8))

for i in range(len(images)):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i], fontsize=9)
    plt.axis('off')

plt.tight_layout()
plt.show()