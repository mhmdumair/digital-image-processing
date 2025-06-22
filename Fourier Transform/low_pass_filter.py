import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1 ─ Read grayscale image
img = cv2.imread("../image/lenna.bmp", cv2.IMREAD_GRAYSCALE)

# 2 ─ Forward 2-D FFT and centre the spectrum
F      = np.fft.fft2(img)
Fshift = np.fft.fftshift(F)
log_spec = 20 * np.log(np.abs(Fshift) + 1)          # log spectrum for display

# 3 ─ Build ideal low-pass mask (radius r)
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
r = 80
mask = np.zeros((rows, cols), np.uint8)
y, x = np.ogrid[:rows, :cols]
mask_area = (y - crow) ** 2 + (x - ccol) ** 2 <= r * r
mask[mask_area] = 1

# 4 ─ Apply mask
Fshift_lp = Fshift * mask
log_spec_lp = 20 * np.log(np.abs(Fshift_lp) + 1)

# 5 ─ Inverse FFT back to spatial domain
img_back = np.fft.ifft2(np.fft.ifftshift(Fshift_lp))
img_back = np.abs(img_back)
img_back = np.clip(img_back, 0, 255).astype(np.uint8)

# 6 ─ Display results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original'); plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(log_spec, cmap='gray')
plt.title('Full Spectrum (log)'); plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(log_spec_lp, cmap='gray')
plt.title('Masked Spectrum (low-pass)'); plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(img_back, cmap='gray')
plt.title('Reconstructed (blurred)'); plt.axis('off')

plt.tight_layout()
plt.show()
