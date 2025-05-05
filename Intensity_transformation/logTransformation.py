import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("../image/meter1.jpg", 0)
if img is None:
    print("Error: Image not found.")
    exit()

print("Image shape:", img.shape)
print("Image dtype:", img.dtype)
print("Min pixel value:", np.min(img))
print("Max pixel value:", np.max(img))


# Log transformation without scaling
log_img = np.log(img + 1)
log_img_array = np.array(log_img, dtype=np.uint8)

# With scaling
scaling_const = 255 / np.log(np.max(img) + 1)
img_with_scaling = scaling_const * np.log(img + 1)
img_with_scaling = np.array(img_with_scaling, dtype=np.uint8)

# Plotting
plt.subplot(221)
plt.imshow(img, cmap="gray")
plt.title("Original")

plt.subplot(222)
plt.imshow(log_img_array, cmap="gray")
plt.title("Log Transformation")

plt.subplot(223)
plt.imshow(img_with_scaling, cmap="gray")
plt.title("With Scaling Constant")

plt.tight_layout()
plt.show()
