import cv2 as cv

img = cv.imread("images/blue.jpg")
cv.imshow("Blue",img)
cv.waitKey(0)