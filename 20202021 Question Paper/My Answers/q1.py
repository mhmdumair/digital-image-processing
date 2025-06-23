import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../Q1_Circles.tif", 0)
img_gray = cv2.GaussianBlur(img, (5, 5), 0)

_, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((42, 42), np.uint8)
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# Find big contours from opened image
img_big = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)  # convert for colored drawing
bigContours, _ = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_big, bigContours, -1, (0, 0, 255), 3)

# Use max contour area, not min, to get big blobs area
cnt = max(bigContours, key=cv2.contourArea)
bigBlobArea = cv2.contourArea(cnt)

# Subtract opened (big blobs) from thresh to isolate small blobs
smallOnlyImage = cv2.subtract(thresh, opened)

# Clean small blobs with morphological opening (better than erosion here)
small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
smallOnlyImage = cv2.morphologyEx(smallOnlyImage, cv2.MORPH_OPEN, small_kernel, iterations=1)

# Find contours in small blobs
contoursAll, _ = cv2.findContours(smallOnlyImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
smallContours = [cnt for cnt in contoursAll if cv2.contourArea(cnt) < bigBlobArea]

# Draw small contours on color image
img_small = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_small, smallContours, -1, (0, 0, 255), 3)

# Plotting
plt.figure(figsize=(10, 20))

print(len(bigContours))
print(len(smallContours))

plt.subplot(4, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(4, 2, 2)
plt.imshow(thresh, cmap="gray")
plt.title('Threshold')
plt.axis('off')

plt.subplot(4, 2, 3)
plt.imshow(opened, cmap="gray")
plt.title('Opened Image (Big blobs)')
plt.axis('off')

plt.subplot(4, 2, 4)
plt.imshow(cv2.cvtColor(img_big, cv2.COLOR_BGR2RGB))
plt.title('Big blobs outlined')
plt.axis('off')

plt.subplot(4, 2, 5)
plt.imshow(smallOnlyImage, cmap="gray")
plt.title('Small blobs only (cleaned)')
plt.axis('off')

plt.subplot(4, 2, 6)
plt.imshow(cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB))
plt.title('Small blobs outlined')
plt.axis('off')

plt.tight_layout()
plt.show()
