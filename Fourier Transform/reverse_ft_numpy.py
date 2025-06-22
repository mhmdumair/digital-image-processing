import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../image/lenna.bmp", 0)

F = np.fft.fft2(img)
fshift_img = np.fft.fftshift(F)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
fshift_img[crow-30 : crow+30, ccol-30 : ccol+30] = 0  # zero centre 60×60

mag_orig  = 20 * np.log(np.abs(np.fft.fftshift(F)) + 1)      # before mask
mag_mask  = 20 * np.log(np.abs(fshift_img) + 1)              # after mask

img_back = np.fft.ifft2(np.fft.ifftshift(fshift_img))
img_back = np.clip(np.abs(img_back), 0, 255).astype(np.uint8)

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray'), plt.title('Original'), plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(mag_orig, cmap='gray'), plt.title('Log |F(u,v)|'), plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(mag_mask, cmap='gray'), plt.title('Masked Spectrum'), plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(img_back, cmap='gray'), plt.title('High-pass Result'), plt.axis('off')

plt.tight_layout()
plt.show()
