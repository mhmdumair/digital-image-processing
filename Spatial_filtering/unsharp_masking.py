import cv2
import matplotlib.pyplot as plt

# Read the image in grayscale
img = cv2.imread("../image/blurry_moon.tif", 0)

gauss1 = cv2.GaussianBlur(img, (11, 11), 0)
sharpened1 = cv2.addWeighted(img, 2, gauss1, -1, 0)

gauss2 = cv2.GaussianBlur(img, (13, 13), 0)
mask = cv2.subtract(img, gauss2)
sharpened2 = cv2.add(img, cv2.multiply(mask, 0.95))

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sharpened1, cmap='gray')
plt.title('Unsharp Masking: Method 1')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sharpened2, cmap='gray')
plt.title('Unsharp Masking: Method 2')
plt.axis('off')

plt.tight_layout()
plt.show()
