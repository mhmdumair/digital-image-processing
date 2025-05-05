import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../image/noisy2.png", 0)

ret,thr1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thr2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
blur = cv2.GaussianBlur(img,(5,5),0)
ret,thr3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.subplot(221)
plt.imshow(img,cmap="gray")
plt.title("Original")

plt.subplot(222)
plt.imshow(thr1,cmap="gray")
plt.title("Global")

plt.subplot(223)
plt.imshow(thr2,cmap="gray")
plt.title("Otsu't thresh holding")

plt.subplot(224)
plt.imshow(thr3,cmap="gray")
plt.title("Otsu's after gaussian blur")

plt.show()

