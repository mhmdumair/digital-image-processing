import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../image/test_pattern_blurring_orig.tif",1)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

avarage = cv2.blur(img,(5,5))
gauss = cv2.GaussianBlur(img,(5,5),0)

plt.subplot(131)
plt.imshow(img)
plt.title("Original")

plt.subplot(132)
plt.imshow(avarage)
plt.title("5 * 5")

plt.subplot(133)
plt.imshow(gauss)
plt.title("Gaussian Burr")

plt.tight_layout()
plt.show()
