import numpy as np
import cv2
import matplotlib.pyplot as plt

COLOR = 'maroon'
FONT_SIZE = 8
KERNEL_SIZE_L = (45,45)
KERNEL_SIZE_S = (31,31)

LARGE_CIRCLE_AREA = 3600
SMALL_CIRCLE_AREA = 1300


img = cv2.imread('images/Q1_Circles.jpg',0)
blur = cv2.medianBlur(img,9)

T1 = 100
ret,inverse_org = cv2.threshold(blur,T1,255,cv2.THRESH_BINARY_INV)

large_cicles = inverse_org[:,260:]

# Elliptical Kernel
kernel_L = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,KERNEL_SIZE_L)
large_cicles_open_ = cv2.morphologyEx(large_cicles, cv2.MORPH_OPEN, kernel_L)

print(large_cicles_open_.dtype)

large_cicles_open_C = cv2.cvtColor(large_cicles_open_,cv2.COLOR_GRAY2BGR)
org_img_c = cv2.cvtColor(inverse_org.copy(),cv2.COLOR_GRAY2BGR)


whole_img_large_only = np.zeros(img.shape,dtype='uint8')
whole_img_large_only[:,260:] =  large_cicles_open_

#ret,whole_img_large_only = cv2.threshold(whole_img_large_only,T1,255,cv2.THRESH_BINARY_INV)

print(whole_img_large_only.dtype)
print(whole_img_large_only)

small_large_removed = inverse_org - whole_img_large_only

kernel_S = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,KERNEL_SIZE_S)
whole_img_small_only = cv2.morphologyEx(small_large_removed, cv2.MORPH_OPEN, kernel_S)


#large counting
image, contours_L, hierarchy = cv2.findContours(whole_img_large_only,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#small counting
image, contours_S, hierarchy = cv2.findContours(whole_img_small_only,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#large counting
large_contours = []

for cnt in contours_L:
    area = cv2.contourArea(cnt)
    print(area)
    if area > (LARGE_CIRCLE_AREA - 200) and area < (LARGE_CIRCLE_AREA + 200): 
        large_contours.append(cnt)
        print('selected contour area : ',area)

print('count_large : ', len(large_contours))

#small counting
small_contours = []

for cnt in contours_S:
    area = cv2.contourArea(cnt)
    print(area)
    if area > (SMALL_CIRCLE_AREA - 50) and area < (SMALL_CIRCLE_AREA + 50): 
        small_contours.append(cnt)
        print('selected contour area : ',area)

print('count_small : ', len(small_contours))

org_img_c_large = cv2.drawContours(org_img_c.copy(), large_contours, -1, (255,0,0), 10)
org_img_c_small = cv2.drawContours(org_img_c.copy(), small_contours, -1, (255,255,0), 5)

all_in_one_ = cv2.drawContours(org_img_c_large.copy(), small_contours, -1, (255,255,0), 5)


counts_image = np.zeros(img.shape,dtype='uint8')
counts_image[:] = 255
print(counts_image)

large_text = "large count = {}".format(len(large_contours))
small_text = "small count = {}".format(len(small_contours))
    

plt.style.use('grayscale')
plt.figure().patch.set_facecolor('white')

plt.subplot(341)
plt.title('Oroginal Image',c=COLOR,fontsize=FONT_SIZE)
plt.axis('off')
plt.imshow(img)

plt.subplot(342)
plt.title('blur',c=COLOR,fontsize=FONT_SIZE)
plt.axis('off')
plt.imshow(blur)

plt.subplot(343)
plt.title('thresh inverse',c=COLOR,fontsize=FONT_SIZE)
plt.axis('off')
plt.imshow(inverse_org)

plt.subplot(344)
plt.title('large_sliced',c=COLOR,fontsize=FONT_SIZE)
plt.axis('off')
plt.imshow(large_cicles)

plt.subplot(345)
plt.title('larger_circles\n(small removed)',c=COLOR,fontsize=FONT_SIZE)
plt.axis('off')
plt.imshow(large_cicles_open_)

plt.subplot(346)
plt.title('whole_img_large_only',c=COLOR,fontsize=FONT_SIZE)
plt.axis('off')
plt.imshow(whole_img_large_only)

plt.subplot(347)
plt.title('smaller_circles\n(large removed)',c=COLOR,fontsize=FONT_SIZE)
plt.axis('off')
plt.imshow(small_large_removed)

plt.subplot(348)
plt.title('whole_img_small_only',c=COLOR,fontsize=FONT_SIZE)
plt.axis('off')
plt.imshow(whole_img_small_only)

plt.subplot(349)
plt.title('larger complete\ncircles - RED',c=COLOR,fontsize=FONT_SIZE)
plt.axis('off')
plt.imshow(org_img_c_large)

plt.subplot(3,4,10)
plt.title('smaller complete\ncircles - YELLOW',c=COLOR,fontsize=FONT_SIZE)
plt.axis('off')
plt.imshow(org_img_c_small)

plt.subplot(3,4,11)
plt.title('all_in_one_',c=COLOR,fontsize=FONT_SIZE)
plt.axis('off')
plt.imshow(all_in_one_)

plt.subplot(3,4,12)
plt.title('COUNTS',c=COLOR,fontsize=FONT_SIZE)
plt.axis('off')
plt.text(35, 220, large_text, fontsize = 10,weight='bold',c='r')
plt.text(35, 390, small_text, fontsize = 10,weight='bold',c='y')
plt.imshow(counts_image,cmap='gray')

plt.show()



