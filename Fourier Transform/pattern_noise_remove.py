import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load grayscale image
img = cv2.imread("../image/clown.jpg", 0)

# 2. Compute 2D FFT and shift zero frequency component to center
ft_img = np.fft.fft2(img)
fshift_img = np.fft.fftshift(ft_img)

# 3. Calculate magnitude spectrum for visualization
magnitude_spectrum_shifted = 20 * np.log(np.abs(fshift_img) + 1)

# 4. Create mask with ones everywhere except small zeros at specific frequencies
rows, cols = img.shape
mask = np.ones((rows, cols), np.uint8)
r = 4  # radius of small circular notch filters

# Coordinates of the frequency components to suppress (notch centers)
center_1, center_2, center_3, center_4 = [39, 106], [51, 42], [88, 21], [75, 84]

x, y = np.ogrid[:rows, :cols]
for center in [center_1, center_2, center_3, center_4]:
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0

# 5. Apply mask to frequency domain (notch filtering)
fshift_and_mask = fshift_img * mask
magnitude_spectrum_masked = 20 * np.log(np.abs(fshift_and_mask) + 1)

# 6. Inverse FFT to reconstruct filtered image
f_ishift = np.fft.ifftshift(fshift_and_mask)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
img_back = np.clip(img_back, 0, 255).astype(np.uint8)

# 7. Display images
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(magnitude_spectrum_shifted, cmap='gray')
plt.title('Magnitude Spectrum')


plt.subplot(2, 2, 3)
plt.imshow(magnitude_spectrum_masked, cmap='gray')
plt.title('Masked Spectrum (Notch Filter)')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(img_back, cmap='gray')
plt.title('Filtered Image (Inverse FFT)')
plt.axis('off')

plt.tight_layout()
plt.show()
