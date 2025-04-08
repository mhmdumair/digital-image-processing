import cv2 as cv

# img = cv.imread("Photos/cat.jpg")
# cv.imshow("Cat",img)
# cv.waitKey(0)

capture = cv.VideoCapture("videos/dog.mp4");     # 0 - webcam  , int : camara number

while True:
    isTrue,frame = capture.read()
    cv.imshow("Video",frame)
    if cv.waitKey(20) and 0xFF == ord('d'):  # if 'd' pressed stop displaying
        break

capture.release()
cv.destroyWindow()

# after last frame it gives error because there is no more frames to display
