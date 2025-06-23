import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../image/star.png")
img1 = img.copy()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

rs, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img1, contours, -1, (0, 255, 0), 2)

cnt = contours[0]
hull_idx = cv2.convexHull(cnt, returnPoints=False)
hull_pnts = cv2.convexHull(cnt)
hull_img = img.copy()

cv2.drawContours(hull_img, [hull_pnts], -1, (0, 0, 255), 2)

defects_img = img.copy()
defects = cv2.convexityDefects(cnt, hull_idx)

defect_points = []  # To store defect coordinates

for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]
    f_point = tuple(cnt[f][0])
    defect_points.append(f_point)
    cv2.circle(defects_img, f_point, 10, (0, 45, 201), -1)

# Print defect points coordinates
print("Convexity defect points (x, y):")
for point in defect_points:
    print(point)

plt.subplot(221)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")

plt.subplot(222)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title("Contours")

plt.subplot(223)
plt.imshow(cv2.cvtColor(hull_img, cv2.COLOR_BGR2RGB))
plt.title("Convex Hull")

plt.subplot(224)
plt.imshow(cv2.cvtColor(defects_img, cv2.COLOR_BGR2RGB))
plt.title("Defects")

plt.show()
