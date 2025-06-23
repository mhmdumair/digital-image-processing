import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image in grayscale
img = cv2.imread("../image/sudoku-original.jpg", 0)

# Sobel Edge Detection
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
sobelxy = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=3)
sobel = cv2.magnitude(sobelx,sobely)

sobelx = np.uint8(cv2.convertScaleAbs(sobelx))
sobely = np.uint8(cv2.convertScaleAbs(sobely))
sobelxy = np.uint8(cv2.convertScaleAbs(sobelxy))

laplacian = cv2.Laplacian(img,cv2.CV_16U,ksize=3)
canny = cv2.Canny(img,100,150)


# Display results
plt.figure(figsize=(8,6))
plt.subplot(3,3,1), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(3,3,2), plt.imshow(sobelx, cmap='gray'), plt.title('SobelX')
plt.subplot(3,3,3), plt.imshow(sobely, cmap='gray'), plt.title('SobelY')
plt.subplot(3,3,4), plt.imshow(sobelxy, cmap='gray'), plt.title('SobelXY')
plt.subplot(3,3,5), plt.imshow(sobel, cmap='gray'), plt.title('Sobel')
plt.subplot(3,3,6), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian')
plt.subplot(3,3,7), plt.imshow(canny, cmap='gray'), plt.title('Canny')
plt.tight_layout()
plt.show()
