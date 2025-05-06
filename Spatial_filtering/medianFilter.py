import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../image/test_pattern_blurring_orig.tif",0)

median = cv2.medianBlur(img,5)

plt.subplot(121)
plt.imshow(img ,cmap="gray")
plt.title("Original")

plt.subplot(122)
plt.imshow(median ,cmap="gray")
plt.title(" median 5 * 5")


plt.tight_layout()
plt.show()
