import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1) Read image as grayscale
img = cv2.imread("../image/lenna.bmp", 0)

# 2) Forward 2-D FFT and shift DC to centre
F      = np.fft.fft2(img)
Fshift = np.fft.fftshift(F)

# 3) Log-scaled magnitude spectrum
mag = 20 * np.log(np.abs(Fshift) + 1)

# 4) High-pass mask: zero a 60×60 square around the centre
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
Fshift_hp = Fshift.copy()
Fshift_hp[crow-30:crow+30, ccol-30:ccol+30] = 0

# 5) Log-scaled magnitude after masking
mag_hp = 20 * np.log(np.abs(Fshift_hp) + 1)

# 6) Inverse FFT back to spatial domain
img_back = np.fft.ifft2(np.fft.ifftshift(Fshift_hp))
img_back = np.clip(np.abs(img_back), 0, 255).astype(np.uint8)

# 7) Display each stage with subplot() and imshow()
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original');              plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(mag, cmap='gray')
plt.title('Log |F(u,v)|');          plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(mag_hp, cmap='gray')
plt.title('Masked Spectrum');       plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(img_back, cmap='gray')
plt.title('High-pass Filtered');    plt.axis('off')

plt.tight_layout()
plt.show()
