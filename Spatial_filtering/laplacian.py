import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../image/sudoku-original.jpg", 0)

# Apply Laplacian with CV_8U
laplacian_8u = cv2.Laplacian(img, cv2.CV_8U, ksize=3)

# Apply Laplacian with CV_16U (unsigned 16-bit)
laplacian_16u = cv2.Laplacian(img, cv2.CV_16U, ksize=3)

# For display: scale the 16U result to 8U for visualization (optional, if needed)
# laplacian_16u_disp = cv2.convertScaleAbs(laplacian_16u)

# Display the results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(laplacian_8u, cmap='gray')
plt.title('Laplacian - CV_8U')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(laplacian_16u, cmap='gray')
plt.title('Laplacian - CV_16U')
plt.axis('off')

plt.tight_layout()
plt.show()
