import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../image/blue.jpg",1)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

kernal1 = np.ones((5,5),np.float32)/25
kernal2 = np.ones((9,9),np.float32)/81

dst = cv2.filter2D(img,-1,kernal1)
dst2 = cv2.filter2D(img,-1,kernal2)

plt.subplot(131)
plt.imshow(img)
plt.title("Original")

plt.subplot(132)
plt.imshow(dst)
plt.title("5 * 5")

plt.subplot(133)
plt.imshow(dst2)
plt.title("9 * 9")

plt.tight_layout()
plt.show()
