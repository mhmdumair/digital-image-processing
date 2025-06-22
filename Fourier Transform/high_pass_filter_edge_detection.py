import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read image (grayscale)
img = cv2.imread("../image/lenna.bmp", 0)

# 2. Apply 2D FFT and shift the DC component to center
ft_img = np.fft.fft2(img)
fshift_img = np.fft.fftshift(ft_img)

# 3. Log-scaled magnitude spectrum for visualization
magnitude_spectrum_shifted = 20 * np.log(np.abs(fshift_img) + 1)

# 4. Create a circular high-pass mask
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
mask = np.ones((rows, cols), np.uint8)
r = 80  # radius of low-frequency area to remove

# Create circular area to suppress (low frequencies)
x, y = np.ogrid[:rows, :cols]
mask_area = (x - crow) ** 2 + (y - ccol) ** 2 <= r*r
mask[mask_area] = 0

# 5. Apply mask to the frequency domain
fshift_and_mask = fshift_img * mask
magnitude_spectrum_masked = 20 * np.log(np.abs(fshift_and_mask) + 1)

# 6. Inverse FFT to return to spatial domain
f_ishift = np.fft.ifftshift(fshift_and_mask)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
img_back = np.clip(img_back, 0, 255).astype(np.uint8)

# 7. Display results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(magnitude_spectrum_shifted, cmap='gray')
plt.title('Original Spectrum (Log)')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(magnitude_spectrum_masked, cmap='gray')
plt.title('Masked Spectrum (High-Pass)')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(img_back, cmap='gray')
plt.title('Reconstructed Image (Edges)')
plt.axis('off')

plt.tight_layout()
plt.show()
