import  cv2
import  numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../harbour.jpg",0)

ftImg = np.fft.fft2(img)
fShift = np.fft.fftshift(ftImg)
fShiftAbs = np.abs(fShift)

magnitude = 20 * np.log(fShiftAbs + 1)
rows,cols = img.shape
mask = np.ones(img.shape,np.uint8)
r = 4

start1 ,end1 = (152,0),(152,rows)
start2 ,end2 = (170,0),(170,rows)

start3 ,end3 = (0,122),(cols,122)
start4 ,end4 = (0,139),(cols,139)

start5 ,end5 = (25,50),(150,123)
start6 ,end6 = (170,137),(301,214)

start7 ,end7 = (23,169),(150,130)
start8 ,end8 = (190,121),(294,91)

cv2.line(mask,start1,end1,(0,0,0),r)
cv2.line(mask,start2,end2,(0,0,0),r)
cv2.line(mask,start3,end3,(0,0,0),r)
cv2.line(mask,start4,end4,(0,0,0),r)
cv2.line(mask,start5,end5,(0,0,0),r)
cv2.line(mask,start6,end6,(0,0,0),r)
cv2.line(mask,start7,end7,(0,0,0),r)
cv2.line(mask,start8,end8,(0,0,0),r)

fShiftAndMask = mask * fShift
flsShift = np.fft.ifftshift(fShiftAndMask)
img_back = np.fft.ifft2(flsShift)
img_back = np.abs(img_back)
img_back = np.uint8(img_back)


plt.subplot(2, 2, 1)
plt.imshow(img , cmap="gray")
plt.title('Small blobs colored')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(magnitude , cmap="gray")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(mask , cmap="gray")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(img_back , cmap="gray")
plt.axis('off')

plt.show()