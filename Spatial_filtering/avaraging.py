import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../image/blue.jpg",1)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

avarage1 = cv2.blur(img,(5,5))
avarage12 = cv2.boxFilter(img,-1,(5,5,))

plt.subplot(131)
plt.imshow(img)
plt.title("Original")

plt.subplot(132)
plt.imshow(avarage1)
plt.title("5 * 5")

plt.subplot(133)
plt.imshow(avarage12)
plt.title("5*5 box filter")

plt.tight_layout()
plt.show()
