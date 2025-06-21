import cv2
import matplotlib.pyplot as plt

# 1) Load the MRI scan (grayscale)
img = cv2.imread("../image/img_mri_brain_tumor.jpg", 0)
if img is None:
    raise FileNotFoundError("Check that the image path is correct!")

# ------------------------------------------------------------------
# PART A – sigmaColor sweep (sigmaSpace fixed = 70)
# ------------------------------------------------------------------
img_50  = cv2.bilateralFilter(img, d=9, sigmaColor=50,  sigmaSpace=70)
img_60  = cv2.bilateralFilter(img, d=9, sigmaColor=60,  sigmaSpace=70)
img_70  = cv2.bilateralFilter(img, d=9, sigmaColor=70,  sigmaSpace=70)
img_100 = cv2.bilateralFilter(img, d=9, sigmaColor=100, sigmaSpace=70)

titles_color = ["Original", "σColor=50", "σColor=60", "σColor=70", "σColor=100"]
images_color = [img, img_50, img_60, img_70, img_100]

plt.figure(figsize=(18, 4))
for i, (im, title) in enumerate(zip(images_color, titles_color), 1):
    plt.subplot(1, 5, i)
    plt.imshow(im, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.suptitle("sigmaColor sweep (σSpace fixed at 70)", fontsize=14)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# PART B – sigmaSpace sweep (sigmaColor fixed = 70)
# ------------------------------------------------------------------
sp_25  = cv2.bilateralFilter(img, d=9, sigmaColor=70, sigmaSpace=25)
sp_50  = cv2.bilateralFilter(img, d=9, sigmaColor=70, sigmaSpace=50)
sp_100 = cv2.bilateralFilter(img, d=9, sigmaColor=70, sigmaSpace=100)

titles_space = ["Original", "σSpace=25", "σSpace=50", "σSpace=100"]
images_space = [img, sp_25, sp_50, sp_100]

plt.figure(figsize=(14, 4))
for i, (im, title) in enumerate(zip(images_space, titles_space), 1):
    plt.subplot(1, 4, i)
    plt.imshow(im, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.suptitle("sigmaSpace sweep (σColor fixed at 70)", fontsize=14)
plt.tight_layout()
plt.show()
