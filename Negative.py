import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("image/meter1.jpg",0)
img1 = cv2.bitwise_not(img)
img2 = 255-img
img3 = img.copy()
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img3[i,j]=255 - img[i,j]

plt.subplot(221)
plt.imshow(img,cmap="gray")
plt.title("Original")

plt.subplot(222)
plt.imshow(img1,cmap="gray")
plt.title("Not")

plt.subplot(223)
plt.imshow(img2,cmap="gray")
plt.title("255-img")

plt.subplot(224)
plt.imshow(img3,cmap="gray")
plt.title("Loop")

plt.show()