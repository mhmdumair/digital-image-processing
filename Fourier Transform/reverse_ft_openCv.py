import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read grayscale image
img = cv2.imread("../image/lenna.bmp", cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# 2. Perform DFT and shift DC to center
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
fshift_img = np.fft.fftshift(dft)

# 3. Create low-pass filter mask (keep center 60x60 region)
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# 4. Apply mask (retain only low frequencies)
fshift_and_mask = fshift_img * mask

# 5. Magnitude spectrum of filtered frequency domain
fshift_and_mask_abs = np.abs(cv2.magnitude(
    fshift_and_mask[:, :, 0], fshift_and_mask[:, :, 1]
))
log_filtered = 20 * np.log(fshift_and_mask_abs + 1)

# 6. Inverse FFT
f_ishift = np.fft.ifftshift(fshift_and_mask)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# 7. Normalize for display
cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
img_back = img_back.astype(np.uint8)

# 8. Display using matplotlib
plt.figure(figsize=(10, 6))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(log_filtered, cmap='gray')
plt.title('Filtered Spectrum')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_back, cmap='gray')
plt.title('Low-pass Result')
plt.axis('off')

plt.tight_layout()
plt.show()
