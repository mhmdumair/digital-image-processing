import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('xray.jpg', 0)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log1p(np.abs(fshift))

rows, cols = img.shape
mask = np.ones((rows, cols), np.uint8)

num_lines = 12
spacing = cols // (num_lines + 1)
line_width = 5
for i in range(1, num_lines + 1):
    mask[:, i * spacing - line_width//2 : i * spacing + line_width//2] = 0

fshift_filtered = fshift * mask
filtered_magnitude_spectrum = np.log1p(np.abs(fshift_filtered))

f_ishift = np.fft.ifftshift(fshift_filtered)
img_denoised = np.abs(np.fft.ifft2(f_ishift))

# Plot the results
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Noisy X-ray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Original Magnitude Spectrum')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(mask * 255, cmap='gray')
plt.title('Denoising Mask')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(filtered_magnitude_spectrum, cmap='gray')
plt.title('Filtered Magnitude Spectrum')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(img_denoised, cmap='gray')
plt.title('Denoised X-ray Image')
plt.axis('off')

plt.tight_layout()
plt.show()
