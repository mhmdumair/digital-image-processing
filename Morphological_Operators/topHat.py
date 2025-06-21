import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread("../image/j.png", cv2.IMREAD_GRAYSCALE)

kernel_1 = np.ones((5, 5), np.uint8)
kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT,   (9, 9))
kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9, 9))
kernel_4 = cv2.getStructuringElement(cv2.MORPH_CROSS,  (9, 9))

topHat1 = cv2.morphologyEx(img,cv2.MORPH_TOPHAT, kernel_1)
topHat2 = cv2.morphologyEx(img,cv2.MORPH_TOPHAT, kernel_2)
topHat3 = cv2.morphologyEx(img,cv2.MORPH_TOPHAT, kernel_3)
topHat4 = cv2.morphologyEx(img,cv2.MORPH_TOPHAT, kernel_4)

images = [img, topHat1, topHat2, topHat3, topHat4]
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
