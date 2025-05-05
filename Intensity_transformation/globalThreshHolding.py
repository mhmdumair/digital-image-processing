import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../image/cameraman.tif", cv2.IMREAD_GRAYSCALE)

T1 = 100

ret, thresh1 = cv2.threshold(img, T1, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, T1, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, T1, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, T1, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, T1, 255, cv2.THRESH_TOZERO_INV)

titles = [
    'Original Image',
    '1. THRESH_BINARY (Document scanning)',
    '2. THRESH_BINARY_INV (Object detection)',
    '3. THRESH_TRUNC (Contrast reduction)',
    '4. THRESH_TOZERO (Feature extraction)',
    '5. THRESH_TOZERO_INV (Background removal)'
]
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

plt.figure(figsize=(12, 8))
plt.suptitle(f"Global Thresholding Comparison (Threshold Value: {T1})", color='blue', fontsize=12)

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i], fontsize=9)
    plt.axis('off')

plt.tight_layout()
plt.show()