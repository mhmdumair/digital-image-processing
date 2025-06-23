import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read and preprocess
img = cv2.imread("../galaxy.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray_smooth = cv2.bilateralFilter(img_gray, 25, 75, 75)  # preserves edges

_, thresh = cv2.threshold(img_gray_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours ,h = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# cnt = max(contours,key = cv2.contourArea)
cnts = [cnt for  cnt in contours if cv2.arcLength(cnt,True)>20]

img_copy = img.copy()
cv2.drawContours(img_copy,[cnts],-1,(0,255,0),2)

# Step 7: Plotting
plt.figure(figsize=(12, 8))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_gray_smooth,cmap="gray")
plt.title('Grayscale Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
plt.title('Grayscale Image')
plt.axis('off')



plt.show()
