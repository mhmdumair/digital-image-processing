import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- load the image in grayscale ---
img = cv2.imread("../image/j.png", cv2.IMREAD_GRAYSCALE)

# --- 5 × 5 structuring elements ---
kernel_1 = np.ones((5, 5), np.uint8)                          # all-ones square
kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT,   (5, 5))
kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
kernel_4 = cv2.getStructuringElement(cv2.MORPH_CROSS,  (5, 5))

open1 = cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel_1)
open2 = cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel_2)
open3 = cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel_3)
open4 = cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel_4)

images = [img, open1, open2, open3, open4]
titles  = ['Original',
           'Ones kernel',
           'Rectangular',
           'Elliptical',
           'Cross-shaped']

plt.figure(figsize=(6, 6))
for idx, (im, ttl) in enumerate(zip(images, titles), start=1):
    plt.subplot(2, 3, idx)
    plt.imshow(im, cmap='gray')
    plt.title(ttl)
    plt.axis('off')

plt.tight_layout()
plt.show()
