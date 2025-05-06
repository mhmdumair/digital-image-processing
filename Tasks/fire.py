import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("../image/fire.jpg", 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mask = img.copy()
mask = cv2.cvtColor(mask, cv2.COLOR_RGB2HSV)

lower_yellow = np.array([13, 50, 180])
upper_yellow = np.array([40, 255, 255])
mask = cv2.inRange(mask, lower_yellow, upper_yellow)

result = cv2.bitwise_and(img,img,mask=mask)


plt.subplot(221)
plt.imshow(img)
plt.title("Original")

plt.subplot(222)
plt.imshow(mask, cmap='gray')

plt.subplot(223)
plt.imshow(result)
plt.title("Result")

plt.tight_layout()
plt.show()
