# import cv2
# import matplotlib.pyplot as plt
# img = cv2.imread("noisy_xray_image.jpg",0)
# gauss = cv2.GaussianBlur(img, (5,5), 0)
# median = cv2.medianBlur(img, 5)
# bil = cv2.bilateralFilter(img, 5, 75, 75)
# plt.subplot(221)
# plt.title("Original")
# plt.imshow(img, cmap="gray")
# plt.axis("off")
# plt.subplot(222)
# plt.title("Gaussian Blur")
# plt.imshow(gauss, cmap="gray")
# plt.axis("off")
# plt.subplot(223)
# plt.title("Median Blur")
# plt.imshow(median, cmap="gray")
# plt.axis("off")
# plt.subplot(224)
# plt.title("Bilateral Filter")
# plt.imshow(bil, cmap="gray")
# plt.axis("off")
# plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('noisy_xray_image.jpg', 0)
f_transform = np.fft.fft2(img)
f_shift = np.fft.fftshift(f_transform)
magnitude_spectrum = 20 * np.log(np.abs(f_shift)+1)
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
mask = np.ones((rows, cols), np.float32)
band_width = img.shape[0]
for i in range(1,9):
 mask[0:band_width, ccol+(i*30)-5:ccol+(i*30)+5] = 0
 mask[0:band_width, ccol-(i*30)-5:ccol-(i*30)+5] = 0
f_shift_filtered = f_shift * mask
magnitude_spectrum_filtered = 20 * np.log(np.abs(f_shift_filtered)+1)
f_ishift = np.fft.ifftshift(f_shift_filtered)
img_denoised = np.abs(np.fft.ifft2(f_ishift))
plt.figure(figsize=(15, 10))
plt.subplot(231)
plt.imshow(img, cmap='gray')
plt.title('Original Noisy X-ray')
plt.axis('off')
plt.subplot(232)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Original Magnitude Spectrum')
plt.axis('off')
plt.subplot(233)
plt.imshow(mask, cmap='gray')
plt.title('Denoising Mask')
plt.axis('off')
plt.subplot(234)
plt.imshow(magnitude_spectrum_filtered, cmap='gray')
plt.title('Filtered Magnitude Spectrum')
plt.axis('off')
plt.subplot(235)
plt.imshow(img_denoised, cmap='gray')
plt.title('Denoised X-ray Image')
plt.axis('off')
plt.show()

