import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('harbour.jpg',0)
blurred_image = cv2.GaussianBlur(img, (43, 43), 0)

plt.subplot(111)

plt.axis('off')
plt.imshow(blurred_image, cmap='gray')
plt.show()
 