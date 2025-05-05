import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../image/cameraman.tif", 0)
img1 = img.copy()

# img1[(img1 < 100) | (img1 > 180)] = 25
img1[(img1 >= 100) & (img1 <= 180)] = 225

# for i in range(img1.shape[0]):
#     for j in range(img1.shape[1]):
#         if img1[i,j] >= 100 and img1[i,j]<=180:
#             img1[i,j] = 225
#         else:
#             img1[i,j] = 25


plt.subplot(121)
plt.imshow(img,cmap="gray")
plt.title("Original")

plt.subplot(122)
plt.imshow(img1,cmap="gray")
plt.title("Gray level slicing")


plt.show()
