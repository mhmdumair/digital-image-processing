import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'image/cameraman.tif', cv2.IMREAD_GRAYSCALE)

# global threshold value
T1 = 100

ret, thresh1 = cv2.threshold(img, T1, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, T1, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, T1, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, T1, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, T1, 255, cv2.THRESH_TOZERO_INV)



titles = ['Original Image', 'THRESH_BINARY', 'THRESH_BINARY_INV',
          'THRESH_TRUNC', 'THRESH_TOZERO', 'THRESH_TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

plt.suptitle("Global Threshold Value : 100", color="blue")
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i], fontsize=7)
    plt.xticks([]), plt.yticks([])

plt.show()


