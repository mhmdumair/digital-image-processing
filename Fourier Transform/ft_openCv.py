import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 1. Read image (grayscale)
# ----------------------------------------------------------
img = cv2.imread("../image/lenna.bmp", cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape

# ----------------------------------------------------------
# 2. Forward DFT (complex output: 2 channels)
# ----------------------------------------------------------
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

# Shift DC to centre for easier masking
dft_shift = np.fft.fftshift(dft, axes=[0, 1])

# ----------------------------------------------------------
# 3. Build a 60×60 high-pass mask (zero low frequencies)
# ----------------------------------------------------------
mask = np.ones((rows, cols, 2), np.uint8)      # 2 channels (real, imag)
crow, ccol = rows // 2, cols // 2
mask[crow-30:crow+30, ccol-30:ccol+30] = 0     # zero out centre square

# ----------------------------------------------------------
# 4. Apply mask in frequency domain
# ----------------------------------------------------------
dft_shift_hp = dft_shift * mask

# Magnitude spectrum *before* and *after* masking (log-scaled)
mag_full = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
mag_hp   = cv2.magnitude(dft_shift_hp[:, :, 0], dft_shift_hp[:, :, 1])
log_full = 20 * np.log(mag_full + 1)
log_hp   = 20 * np.log(mag_hp   + 1)


f_ishift = np.fft.ifftshift(dft_shift_hp, axes=[0, 1])
# Inverse transform
img_back = cv2.idft(f_ishift)
# Take real magnitude
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
# Normalise to 0–255 for display
cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
img_back = img_back.astype(np.uint8)

# ----------------------------------------------------------
# 6. Display every step with subplot() and imshow()
# ----------------------------------------------------------
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original');            plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(log_full, cmap='gray')
plt.title('Log |F(u,v)|');        plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(log_hp, cmap='gray')
plt.title('Masked Spectrum');     plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(img_back, cmap='gray')
plt.title('High-pass Result');    plt.axis('off')

plt.tight_layout()
plt.show()
