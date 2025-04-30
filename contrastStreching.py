import numpy as np
import matplotlib.pyplot as plt
import cv2

# Read image and verify
img = cv2.imread("image/contrast_str.png", 0)
if img is None:
    raise ValueError("Image not loaded - check path or file")

img_max = np.max(img)
img_min = np.min(img)

output_img = (img - img_min) * (255.0/(img_max - img_min))
output_img = np.clip(output_img, 0, 255).astype(np.uint8)

plt.figure(figsize=(12, 10))


# Original Image
plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis('off')

# Original Histogram
plt.subplot(2, 2, 2)
plt.hist(img.ravel(), bins=256, range=(0,256), color='blue', alpha=0.7)
plt.title("Original Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

# Output Image
plt.subplot(2, 2, 3)
plt.imshow(output_img, cmap="gray")
plt.title("Contrast Stretched Image")
plt.axis('off')

# Output Histogram
plt.subplot(2, 2, 4)
plt.hist(output_img.ravel(), bins=256, range=(0,256), color='red', alpha=0.7)
plt.title("Stretched Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
