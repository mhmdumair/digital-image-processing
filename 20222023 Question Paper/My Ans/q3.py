import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_vri(img):
    B,G,R = cv2.split(img)
    vri = (G - R) / (G + R - B - 0.1)
    return vri


def normalize_vri(img):
    normalized = ((img+1)*127.5)
    return np.uint8(normalized)

def apply_threshhold_colormap(img,thresh):
    img1 = np.ones((img.shape[0],img.shape[0],3) , np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j]<thresh:
                img1[i,j] = [255,0,0]
            else:
                img1[i, j] = [0,255, 0]

    return img1

img = cv2.imread("../land.jpg")
print(img.shape)
img_vri = calculate_vri(img)
print(img_vri.shape)
normalize = normalize_vri(img_vri)
print(normalize.shape)
img_out = apply_threshhold_colormap(normalize,100)
print(img_out.shape)

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_out,cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.savefig("color_map.jpg")
plt.show()

