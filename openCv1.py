import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1 = cv.imread("image/messi.jpg",0)
img2 = cv.imread("image/OpenCV_Logo.jpg",1)
# cv.imshow("Image",img1)

# Horizontal Stack
# cv.namedWindow("Main",cv.WINDOW_NORMAL)
# cv.imshow("Main",np.hstack((img1,img1)))

# Vertical Stack
# cv.imshow("Main",np.vstack((img1,img1)))

# cv.waitKey()
# cv.destroyWindow()


# Display Images Using Matplotlib

# img1 = cv.cvtColor(img1,cv.COLOR_BGR2RGB)
# plt.imshow(img1)
# plt.xticks([])
# plt.yticks([])
# plt.savefig("image/test.jpg")
# plt.show()
#
# cv.imwrite("image/messi_gray.jpg",img1)

# imgr = cv.resize(img1,(500,500))
# img1 = cv.cvtColor(img1,cv.COLOR_BGR2RGB)
# plt.subplot(121)
# plt.imshow(img1)
# plt.title("Original")
#
# plt.subplot(122)
# imgr = cv.cvtColor(imgr,cv.COLOR_BGR2RGB)
# plt.imshow(imgr)
# plt.title("Resized")
#
# plt.show()


# imgr = cv.applyColorMap(img1,cv.COLORMAP_OCEAN)
# imgr = cv.cvtColor(imgr,cv.COLOR_BGR2RGB)
# img1 = cv.cvtColor(img1,cv.COLOR_BGR2RGB)
# plt.subplot(121)
# plt.imshow(img1)
# plt.title("Original")
#
# plt.subplot(122)
# plt.imshow(imgr)
# plt.title("Resized")
#
# plt.show()


# find HSV values for RGB

green = np.uint8([[[0,255,0]]])
red = np.uint8([[[0,0,255]]])
blue = np.uint8([[[255,0,0]]])

hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
hsv_red = cv.cvtColor(red,cv.COLOR_BGR2HSV)
hsv_blue =  cv.cvtColor(blue,cv.COLOR_BGR2HSV)
print(hsv_green)




