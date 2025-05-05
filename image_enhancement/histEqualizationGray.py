import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../image/Aerial.tif", 0)
img1 = cv2.equalizeHist(img)

plt.subplot(221)
plt.imshow(img,cmap="gray")
plt.title("Original Image")

plt.subplot(222)
plt.hist(img.ravel(), bins=255, range=(0,255))
plt.title("Original Image")

plt.subplot(223)
plt.imshow(img1,cmap="gray")
plt.title("Enhanced Image")

plt.subplot(224)
plt.hist(img1.ravel(), bins=255, range=(0,255))
plt.title("Enhanced hist")

plt.show()