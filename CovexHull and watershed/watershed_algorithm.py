import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 0.  Image path — change this to your own file
# ------------------------------------------------------------
IMG_PATH = "../image/water_coins.jpg"   # <-- adjust as needed
img      = cv2.imread(IMG_PATH)

if img is None:
    raise FileNotFoundError(f"Could not read file: {IMG_PATH}")

# ------------------------------------------------------------
# 1.  Pre-processing: gray conversion
# ------------------------------------------------------------
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ------------------------------------------------------------
# 2.  Global binary mask (Otsu + inverted)
# ------------------------------------------------------------
_, thresh = cv2.threshold(img_gray, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# ------------------------------------------------------------
# 3.  Morphological opening (noise removal)
# ------------------------------------------------------------
kernel  = np.ones((3, 3), np.uint8)
opened  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# ------------------------------------------------------------
# 4.  “Sure” background (dilate)
# ------------------------------------------------------------
sure_bg = cv2.dilate(opened, kernel, iterations=3)

# ------------------------------------------------------------
# 5.  Distance transform + “sure” foreground
# ------------------------------------------------------------
dist_transform = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
_, sure_fg     = cv2.threshold(dist_transform,
                               0.7 * dist_transform.max(), 255,
                               cv2.THRESH_BINARY)
sure_fg = sure_fg.astype(np.uint8)

# ------------------------------------------------------------
# 6.  Unknown region = sure_bg – sure_fg
# ------------------------------------------------------------
unknown = cv2.subtract(sure_bg, sure_fg)

# ------------------------------------------------------------
# 7.  Connected components to label foreground seeds
# ------------------------------------------------------------
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1            # reserve label 1 for background
markers[unknown == 255] = 0      # unknown → 0

# ------------------------------------------------------------
# 8.  Watershed segmentation
# ------------------------------------------------------------
markers = cv2.watershed(img, markers)
segmented = img.copy()
segmented[markers == -1] = (255, 0, 0)   # boundary pixels → red

# ------------------------------------------------------------
# 9.  Display everything with Matplotlib
# ------------------------------------------------------------
steps  = [
    ("Original",          cv2.cvtColor(img, cv2.COLOR_BGR2RGB), None),
    ("Binary (Otsu INV)", thresh,       "gray"),
    ("Opened",            opened,       "gray"),
    ("Sure background",   sure_bg,      "gray"),
    ("Distance transform",dist_transform, "gray"),
    ("Sure foreground",   sure_fg,      "gray"),
    ("Unknown",           unknown,      "gray"),
    ("Watershed result",  cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB), None)
]

coin_labels = np.unique(markers)
coin_labels = coin_labels[coin_labels>1]
print(coin_labels)
print(len(coin_labels))

plt.figure(figsize=(10, 12))
for i, (title, im, cmap) in enumerate(steps, 1):
    plt.subplot(4, 2, i)
    plt.imshow(im, cmap=cmap)
    plt.title(title)
    plt.axis("off")
plt.tight_layout()
plt.show()
