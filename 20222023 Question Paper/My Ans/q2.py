import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../planet_surface.png")
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ffImg = np.fft.fft2(img_gray)
fShiftImg = np.fft.fftshift(ffImg)
fShiftImgAbs = np.abs(fShiftImg)

spectrum = 20 * np.log(fShiftImgAbs + 1)

rows,cols = img_gray.shape
mask = np.ones((rows,cols),np.uint8)
r = 4

x,y = np.ogrid[:rows,:cols]
c1, c2,c3,c4 = [222,216],[222,385],[398,214],[398,384]
for c in [c1,c2,c3,c4]:
    maskArea = (x-c[0])**2 + (y-c[1])**2 <= r**r
    mask[maskArea] = 0

r = 3
start1,end1 = (297,0),(297,268)
start2,end2 = (297,349),(297,rows)

start3,end3 = (0,311),(263,311)
start4,end4 = (332,311),(cols,311)



cv2.line(mask,start1,end1,(0,0,0),r)
cv2.line(mask,start2,end2,(0,0,0),r)
cv2.line(mask,start3,end3,(0,0,0),r)
cv2.line(mask,start4,end4,(0,0,0),r)

fftShitAndMask = mask*fShiftImg
# fftShitAndMask = np.abs(fftShitAndMask)
flsShift = np.fft.ifftshift(fftShitAndMask)
img_back = np.fft.ifft2(flsShift)
img_back = np.clip(img_back,0,255).astype(np.uint8)

cv2.imwrite("planet_serface_denoised.jpg",img_back)

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(spectrum,cmap="gray")
plt.title('Grayscale Image')
plt.axis('off')
#

plt.subplot(2, 2, 3)
plt.imshow(mask,cmap="gray")
plt.title('Grayscale Image')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(img_back , cmap="gray")
plt.title('Grayscale Image')
plt.axis('off')

plt.savefig("planet_serface_denoised_steps.png", dpi=300, bbox_inches='tight')
plt.show()
