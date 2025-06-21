import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../image/overlap_coins.jpg")
img1 = img.copy()

img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
img_gray = cv2.morphologyEx(img_gray,cv2.MORPH_OPEN,se)

se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
img_gray = cv2.morphologyEx(img_gray,cv2.MORPH_CLOSE,se)

se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
img_gray = cv2.erode(img_gray,se)

rs,thresh = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img1,contours,-1,(0,255,0),2)

print(len(contours))

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)


plt.subplot(131)
plt.imshow(img)

plt.subplot(132)
plt.imshow(img_gray,cmap="gray")

plt.subplot(133)
plt.imshow(img1)

plt.show()