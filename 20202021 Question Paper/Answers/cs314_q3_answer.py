import numpy as np
import cv2
import matplotlib.pyplot as plt

COLOR = 'maroon'
FONT_SIZE = 8

img_1 = cv2.imread('../messi_1.jpg')
img_2 = cv2.imread('../messi_1.jpg')

img_1_ = cv2.cvtColor(img_1,cv2.COLOR_BGR2RGB)
img_2_ = cv2.cvtColor(img_2,cv2.COLOR_BGR2RGB)

def increase_brightness_color(img, value=80):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    #handling value overflow
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
     
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)

    return img

img_1_en = increase_brightness_color(img_1)
img_2_en = increase_brightness_color(img_2)

plt.subplot(141)
plt.title('Original Image 1',c=COLOR,fontsize=FONT_SIZE)
plt.axis('off')
plt.imshow(img_1_)

plt.subplot(142)
plt.title('Img 1 enhanced',c=COLOR,fontsize=FONT_SIZE)
plt.axis('off')
plt.imshow(img_1_en)

plt.subplot(143)
plt.title('Original Image 2',c=COLOR,fontsize=FONT_SIZE)
plt.axis('off')
plt.imshow(img_2_)

plt.subplot(144)
plt.title('Img 2 enhanced',c=COLOR,fontsize=FONT_SIZE)
plt.axis('off')
plt.imshow(img_2_en)

plt.show()



