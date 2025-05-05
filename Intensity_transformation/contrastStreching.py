import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../image/contrast_str.png", 0)

input_max = np.max(img)
input_min = np.min(img)

if input_max == input_min:
    print("Image is uniform, contrast stretching has no effect")
    output_img = img.copy()
else:
    output_img = 0 + (img - input_min) * (255.0-0)/(input_max-input_min)
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)

plt.subplot(221)
plt.imshow(img,cmap="gray")
plt.title("Original")

plt.subplot(222)
plt.hist(img.ravel(),bins=256,range=(0,256))
plt.title("Original")

plt.subplot(223)
plt.imshow(output_img,cmap="gray")
plt.title("Contrast streched")

plt.subplot(224)
plt.hist(output_img.ravel(),bins=256,range=(0,256))
plt.title("Contrast hist")

plt.show()


