import cv2
import numpy as np
import matplotlib.pyplot as plt

xray = cv2.imread("image/noisy_xray_image.jpg",0)

f = np.fft.fft2(xray)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
rows, cols = xray.shape
mask = np.ones((rows, cols), dtype=np.uint8)
r = 6
center_1 = [338, 240]
center_2 = [339, 270]
center_3 = [336, 332]
center_4 = [338, 361]

x, y = np.ogrid[:rows, :cols]
for cx, cy in [center_1, center_2,center_3,center_4]:
    mask_area = (x - cx)**2 + (y - cy)**2 <= r**2
    mask[mask_area] = 0
fshift_filtered = fshift * mask
magnitude_filtered = 20 * np.log(np.abs(fshift_filtered)
+ 1)
f_ishift = np.fft.ifftshift(fshift_filtered)
img_back = np.abs(np.fft.ifft2(f_ishift))
img_back_normalized = cv2.normalize(img_back, None, 0,
255, cv2.NORM_MINMAX)
img_back_uint8 = np.uint8(img_back_normalized)

plt.subplot(2, 3, 1), plt.imshow(xray, cmap='gray'),
plt.title("Original")
plt.subplot(2, 3, 2), plt.imshow(magnitude_spectrum,
cmap='gray'), plt.title("FFT Magnitude (Before)")
plt.subplot(2, 3, 3), plt.imshow(mask * 255,
cmap='gray'), plt.title("Band Reject Mask")
plt.subplot(2, 3, 4), plt.imshow(magnitude_filtered,
cmap='gray'), plt.title("FFT Magnitude (After)")
plt.subplot(2, 3, 5), plt.imshow(img_back_uint8,
cmap='gray'), plt.title("Denoised Image")
plt.tight_layout()

plt.show()