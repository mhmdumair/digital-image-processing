import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../image/1.jpg", 1)

img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(img)

v_  = cv2.equalizeHist(v)
img1 = cv2.merge((h,s,v_))

img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
img1 = cv2.cvtColor(img1,cv2.COLOR_HSV2RGB)


plt.subplot(221)
plt.imshow(img)
plt.title("Original Image")

plt.subplot(222)
plt.hist(v.ravel(), bins=255, range=(0,255))
plt.title("Original Image")

plt.subplot(223)
plt.imshow(img1)
plt.title("Enhanced Image")

plt.subplot(224)
plt.hist(v_.ravel(), bins=255, range=(0,255))
plt.title("Enhanced hist")

plt.show()