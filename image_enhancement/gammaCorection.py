# p = 255 * (p/255)
# gamma < 1 brighten the image
# gamma > 1 darken the image


import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../image/graylevel6.jpg",0)

def gamma(img, gamma):
    normalized = img / 255.0
    gam = 255.0 * (normalized ** gamma)
    return np.array(gam).astype(np.uint8)

enhanced2 = gamma(img,0.2)
enhanced5 = gamma(img,0.5)
enhanced10 = gamma(img,1)
enhanced20 = gamma(img,2.0)



plt.subplot(321)
plt.imshow(img,cmap="gray")
plt.title("Original")

plt.subplot(322)
plt.imshow(enhanced2,cmap="gray")
plt.title("Gamma 0.2")

plt.subplot(323)
plt.imshow(enhanced5,cmap="gray")
plt.title("Gamma 0.5")

plt.subplot(324)
plt.imshow(enhanced10,cmap="gray")
plt.title("Gamma 1")

plt.subplot(325)
plt.imshow(enhanced20,cmap="gray")
plt.title("Gamma 2")


plt.show()