import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('cell_segmentation.jpg', 0)
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY +
cv2.THRESH_OTSU)
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(),
255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
markers = cv2.watershed(img_color, markers)
img_color[markers == -1] = [255,255,255]
cell_count = len(np.unique(markers)) - 2
print("Number of cells detected:", cell_count)
plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')
plt.subplot(132)
plt.imshow(markers)
plt.title('Watershed Markers')
plt.axis('off')
plt.subplot(133)
plt.imshow(img_color)
plt.title('Segmented Cells')
plt.axis('off')
plt.show()
