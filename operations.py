import numpy as np
import matplotlib.pyplot as plt
import cv2

        # IMAGE RESIZE

# img = cv2.imread("image/3.jpg",1)
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# resized = cv2.resize(img,(500,500))
#
# plt.subplot(121)
# plt.imshow(img)
# plt.title("Original Image")
#
# plt.subplot(122)
# plt.imshow(resized)
# plt.title("Resized Image")
#
# plt.show()


img = cv2.imread("image/3.jpg",1)
b,g,r = cv2.split(img)

b_ = b.copy()
b_[:,:] = 0
image_Blue_removed = cv2.merge((b_,g,r))

g_ = g.copy()
g_[:,:] = 0
image_Green_removed = cv2.merge((b,g_,r))

r_ = r.copy()
r_[:,:] = 0
image_Red_removed = cv2.merge((b,g,r_))

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
blue_removed_rgb = cv2.cvtColor(image_Blue_removed, cv2.COLOR_BGR2RGB)
green_removed_rgb = cv2.cvtColor(image_Green_removed, cv2.COLOR_BGR2RGB)
red_removed_rgb = cv2.cvtColor(image_Red_removed, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 8))

plt.subplot(221)
plt.imshow(img_rgb)
plt.title("Original Image")

plt.subplot(222)
plt.imshow(blue_removed_rgb)
plt.title("Blue Removed")

plt.subplot(223)
plt.imshow(green_removed_rgb)
plt.title("Green Removed")

plt.subplot(224)
plt.imshow(red_removed_rgb)
plt.title("Red Removed")

plt.show()