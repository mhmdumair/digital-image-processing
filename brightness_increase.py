import numpy as np
import matplotlib.pyplot as plt
import cv2

        # GRAY SCALE IMAGE

# def increase_brightness(img,value):
#     img1 = img.copy()
#     limit = 255 - value
#     img1[img1>limit] = 255
#     img1[img1<= limit] += value
#     return img1
#
# img = cv2.imread("image/graylevel6.jpg",0)
# img1 = img.copy()
# img1 = cv2.add(img1,100)
#
# img2 = increase_brightness(img,100)
#
# plt.subplot(131)
# plt.imshow(img,cmap="gray")
# plt.title("Original")
#
# plt.subplot(132)
# plt.imshow(img1,cmap="gray")
# plt.title("Add")
#
# plt.subplot(133)
# plt.imshow(img2,cmap="gray")
# plt.title("Function")
#
#
# plt.show()

def increase_brightness_color_1(img,value):
    img1 = img.copy()
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img1)
    limit = 255 - value
    v[v>limit] = 255
    v[v<= limit] += value
    img1 = cv2.merge((h,s,v))
    img1 = cv2.cvtColor(img1, cv2.COLOR_HSV2BGR)
    return img1

def increase_brightness_color_2(img,value):
    img1 = img.copy()
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            limit = 255-value
            if img1[i,j,2] > limit:
                img1[i, j, 2] = 255
            else:
                img1[i, j, 2] += value
    img1 = cv2.cvtColor(img1, cv2.COLOR_HSV2BGR)
    return img1

img = cv2.imread("image/messi5.jpg",1)
img1 = increase_brightness_color_1(img,80)
img2 = increase_brightness_color_2(img,80)

plt.subplot(131)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title("Original")

plt.subplot(132)
plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
plt.title("Split")

plt.subplot(133)
plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
plt.title("Loop")


plt.show()


