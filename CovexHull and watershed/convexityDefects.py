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
hull_idx = convexHull = cv2.convexHull(cnt,returnPoints=False)
hull_pnts = convexHull = cv2.convexHull(cnt)
hull_img = img.copy()

cv2.drawContours(hull_img,[hull_pnts],-1,(0,0,255),2)

defects_img = img.copy()
defects = cv2.convexityDefects(cnt,hull_idx)

for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]
    f_point = tuple(cnt[f][0])
    cv2.circle(defects_img, f_point, 10, (0,45,201), -1)

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
plt.subplot(224)
plt.imshow(defects_img)
plt.title("Defects")

plt.show()