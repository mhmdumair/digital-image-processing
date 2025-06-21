import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread("../image/j.png", cv2.IMREAD_GRAYSCALE)


kernel_1 = np.ones((5, 5), np.uint8)
kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT,   (5, 5))
kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
kernel_4 = cv2.getStructuringElement(cv2.MORPH_CROSS,  (5, 5))

erosion1 = cv2.erode(img, kernel_1, iterations=1)
erosion2 = cv2.erode(img, kernel_2, iterations=1)
erosion3 = cv2.erode(img, kernel_3, iterations=1)
erosion4 = cv2.erode(img, kernel_4, iterations=1)


images = [img, erosion1, erosion2, erosion3, erosion4]
titles  = ['Original',
           'Ones kernel',
           'Rectangular',
           'Elliptical',
           'Cross‑shaped']

plt.figure(figsize=(15, 3))
for idx, (im, ttl) in enumerate(zip(images, titles), start=1):
    plt.subplot(1, 5, idx)
    plt.imshow(im, cmap='gray')
    plt.title(ttl)
    plt.axis('off')

plt.tight_layout()
plt.show()
