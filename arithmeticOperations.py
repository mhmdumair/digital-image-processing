import numpy as np
import matplotlib.pyplot as plt
import cv2

img1 = cv2.imread("image/messi.jpg",1)
img2 = cv2.imread("image/mask_single.jpg",1)

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2 = cv2.resize(img2,(620,420))

# output = cv2.add(img1,img2)
# output = cv2.addWeighted(img1,0.2,img2,0.8,0)
# output = cv2.subtract(img1,img2)
# output = cv2.multiply(img1,img2)
output =  cv2.divide(img1,img2)

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
