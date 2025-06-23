import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load grayscale image
img = cv2.imread("harbour.jpg", 0)

# 2. Compute 2D FFT and shift zero frequency component to center
ft_img = np.fft.fft2(img)
fshift_img = np.fft.fftshift(ft_img)

# 3. Calculate magnitude spectrum for visualization
magnitude_spectrum_shifted = 20 * np.log(np.abs(fshift_img) + 1)

# 4. Create mask with ones everywhere except small zeros at specific frequencies
rows, cols = img.shape
mask = np.ones((rows, cols), np.uint8)
r = 2  # radius of small circular notch filters

# Coordinates of the frequency components to suppress (notch centers)
# center_1, center_2= [122, 150], [137, 169]
#
# x, y = np.ogrid[:rows, :cols]
# for center in [center_1, center_2]:
#     mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
#     mask[mask_area] = 0

start1,end1 = (150,0),(150,rows)
start2,end2 = (169,0),(169,rows)
start3,end3 = (0,121),(cols,121)
start4,end4 = (0,137),(cols,137)

start5,end5 = (15,170),(132,136)
start6,end6 = (188,122),(317,84)

start7,end7 = (150,122),(20,45)
start8,end8 = (170,136),(296,211)

cv2.line(mask,start1,end1,(0,0,0),r)
cv2.line(mask,start2,end2,(0,0,0),r)
cv2.line(mask,start3,end3,(0,0,0),r)
cv2.line(mask,start4,end4,(0,0,0),r)
cv2.line(mask,start5,end5,(0,0,0),r)
cv2.line(mask,start6,end6,(0,0,0),r)
cv2.line(mask,start7,end7,(0,0,0),r)
cv2.line(mask,start7,end7,(0,0,0),r)

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
