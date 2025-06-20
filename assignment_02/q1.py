import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image in grayscale
img = cv2.imread("../image/sudoku-original.jpg", 0)

# Sobel Edge Detection
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
sobel = cv2.magnitude(sobelx, sobely)

# Laplacian Edge Detection
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# Canny Edge Detection
canny = cv2.Canny(img, 100, 200)

# Display results
plt.figure(figsize=(8,6))
plt.subplot(2,2,1), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(2,2,2), plt.imshow(sobel, cmap='gray'), plt.title('Sobel')
plt.subplot(2,2,3), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian')
plt.subplot(2,2,4), plt.imshow(canny, cmap='gray'), plt.title('Canny')
plt.tight_layout()
plt.show()
