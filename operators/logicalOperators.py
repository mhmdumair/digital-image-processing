import numpy as np
import matplotlib.pyplot as plt
import cv2

img1 = cv2.imread("../image/3.jpg", 1)
img2 = cv2.imread("../image/mask_single.jpg", 1)

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2 = cv2.resize(img2,(1600,1200))

# output =  cv2.bitwise_and(img1,img2)
# output = cv2.bitwise_or(img1,img2)
# output = cv2.bitwise_not(img1)
output = cv2.bitwise_xor(img1,img2)

plt.figure(figsize=(10, 5))

plt.subplot(131)
plt.imshow(img1)
plt.title("Image1")

plt.subplot(132)
plt.imshow(img2)
plt.title("Image2")

plt.subplot(133)
plt.imshow(output)
plt.title("Output")

plt.tight_layout()
plt.show()
