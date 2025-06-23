import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("cell_segmentation.jpg")

img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

rs,thresh = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernal = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernal,iterations=1)
sureBg = cv2.dilate(opening,kernal,iterations=1)

distTransform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sureFg = cv2.threshold(distTransform,0.5*distTransform.max(),255,0)
sureFg = np.uint8(sureFg)

unknown = np.subtract(sureBg ,sureFg)

_,markers = cv2.connectedComponents(sureFg)
markers+=1

markers[unknown==255] = 0
markers = cv2.watershed(img,markers)

segmented = img.copy()
segmented[markers==-1] = (0,0,255)

cellLabel = np.unique(markers)
cellLabel = cellLabel[cellLabel>1]
print(len(cellLabel))


plt.figure(figsize=(8,6))
plt.subplot(2,3,1), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(2,3,2), plt.imshow(sureBg, cmap='gray'), plt.title("Sure background")
plt.subplot(2,3,3), plt.imshow(sureFg, cmap='gray'), plt.title('Sure fore ground')
plt.subplot(2,3,4), plt.imshow(unknown, cmap='gray'), plt.title('Unknown')
plt.subplot(2,3,5), plt.imshow(segmented, cmap='gray'), plt.title('Segmented')

plt.show()