import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("../image/meter1.jpg", 0)

img_float = img.astype(np.float64)

c = 255.0 / np.log(1 + np.max(img_float))
log_img_scaled = c * np.log(img_float+1)  # log1p is more accurate for log(1+x)
log_image2 = np.clip(log_img_scaled, 0, 255).astype(np.uint8)  # Clip values before conversion

log_img = np.log(img_float+1)  # Use log1p for numerical stability
log_image1 = np.clip(log_img, 0, 255).astype(np.uint8)

print(log_image1[100:200, 100:200])
print(log_image2[100:200,100:200])

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(log_image1, cmap='gray')
plt.title("Normalized Log")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(log_image2, cmap='gray')
plt.title("Scaled Log")
plt.axis("off")

plt.tight_layout()
plt.show()
