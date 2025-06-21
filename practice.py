import cv2
import matplotlib.pyplot as plt

# Read the image in color
img = cv2.imread("image/plus.jpg", 1);
img1 = img.copy()
img2 = img.copy()

img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
rs,thresh = cv2.threshold(img_gray,50,255,cv2.THRESH_BINARY)
contours ,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img1,contours,-1,(0,255,0),1)
cv2.drawContours(img2,[contours[2],contours[3]],-1,(255,0,0),2)
plt.figure(figsize=(10, 6))

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

print(len(contours))

plt.subplot(131)
plt.imshow(img)

plt.subplot(132)
plt.imshow(img1)

plt.subplot(133)
plt.imshow(img2)

plt.show()

