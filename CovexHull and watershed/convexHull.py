import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../image/star.png")
img1 = img.copy()

img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
rs , thresh = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

contoures , _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE,)

cv2.drawContours(img1,contoures,-1,(0,255,0),2)

cnt =contoures[0]
hull = convexHull = cv2.convexHull(cnt)
hull_img = img.copy()

cv2.drawContours(hull_img,[hull],-1,(0,0,255),2)

plt.subplot(221)
plt.imshow(img)
plt.title("Original")

plt.subplot(222)
plt.imshow(img1)
plt.title("contoure")

plt.subplot(223)
plt.imshow(hull_img)
plt.title("Convex hull")
#
# plt.subplot(224)
# plt.imshow(thr3,cmap="gray")
# plt.title("Otsu's after gaussian blur")

plt.show()