import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("noisy_xray_image.jpg",0)

# blur = cv2.blur(img,(5,5))
# gaussianBlurr = cv2.GaussianBlur(img,(5,5),0)
# median = cv2.medianBlur(img,5)
# bilateral = cv2.bilateralFilter(img,9,50,50)
#
# plt.figure(figsize=(8,6))
# plt.subplot(2,3,1), plt.imshow(img, cmap='gray'), plt.title('Original')
# plt.subplot(2,3,2), plt.imshow(blur, cmap='gray'), plt.title('Avaraging')
# plt.subplot(2,3,3), plt.imshow(gaussianBlurr, cmap='gray'), plt.title('Gaussian Blurr')
# plt.subplot(2,3,4), plt.imshow(median, cmap='gray'), plt.title('Median')
# plt.subplot(2,3,5), plt.imshow(bilateral, cmap='gray'), plt.title('Bilateral')
#
# plt.tight_layout()
# plt.show()

ftImg = np.fft.fft2(img)
fShifting = np.fft.fftshift(ftImg)
fShiftingAbs = np.abs(fShifting)

spectrum = 20 * np.log(fShiftingAbs + 1)
rows,cols = img.shape
mask = np.ones((rows,cols),np.uint8)
r = 4

c1,c2,c3,c4 = [335,269],[336,330],[335,361],[336,240]
cenetrs = [c1,c2,c3,c4]

x,y = np.ogrid[:rows,:cols]

for c in cenetrs:
   maskArea = (x-c[0])**2 + (y-c[1])**2 <= r*r
   mask[maskArea] = 0

fShiftAndMask =  mask * fShifting
magnitudeSpectrumMasked = 20* np.log(np.abs(fShiftAndMask)+1)
flsShift = np.fft.ifftshift(fShiftAndMask)
img_back = np.fft.ifft2(flsShift)
img_back = np.abs(img_back)

plt.figure(figsize=(8,6))
plt.subplot(2,3,1), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(2,3,2), plt.imshow(spectrum, cmap='gray'), plt.title('Magnitude spectrum')
plt.subplot(2,3,3), plt.imshow(mask, cmap='gray'), plt.title('Mask')
plt.subplot(2,3,4), plt.imshow(magnitudeSpectrumMasked, cmap='gray'), plt.title('After mask')
plt.subplot(2,3,5), plt.imshow(img_back, cmap='gray'), plt.title('Noise removed')


plt.tight_layout()
plt.show()