import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../image/clahe_1.jpg",0)

img_hist = cv2.equalizeHist(img)
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
img_hist_clahe = clahe.apply(img)

plt.subplot(221)
plt.imshow(img,cmap="gray")
plt.title("Original Image")

plt.subplot(222)
plt.imshow(img_hist,cmap="gray")
plt.title("Without CLAHE")

plt.subplot(223)
plt.imshow(img_hist_clahe,cmap="gray")
plt.title("CLAHE")


plt.show()