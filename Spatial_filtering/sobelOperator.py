import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../image/sudoku-original.jpg", 0)

# Compute Sobel for all depths
grad_x_8u = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
grad_y_8u = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)
grad_xy_8u = grad_x_8u + grad_y_8u

grad_x_32f = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
grad_y_32f = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
grad_xy_32f = grad_x_32f + grad_y_32f

grad_x_64f = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
grad_y_64f = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
grad_xy_64f = grad_x_64f + grad_y_64f

# For display, convert float images to uint8
grad_x_32f_disp = cv2.convertScaleAbs(grad_x_32f)
grad_y_32f_disp = cv2.convertScaleAbs(grad_y_32f)
grad_xy_32f_disp = cv2.convertScaleAbs(grad_xy_32f)

grad_x_64f_disp = cv2.convertScaleAbs(grad_x_64f)
grad_y_64f_disp = cv2.convertScaleAbs(grad_y_64f)
grad_xy_64f_disp = cv2.convertScaleAbs(grad_xy_64f)

plt.figure(figsize=(15, 9))

# CV_8U
plt.subplot(3, 3, 1), plt.imshow(grad_x_8u, cmap='gray'), plt.title('Sobel X - CV_8U'), plt.axis('off')
plt.subplot(3, 3, 2), plt.imshow(grad_y_8u, cmap='gray'), plt.title('Sobel Y - CV_8U'), plt.axis('off')
plt.subplot(3, 3, 3), plt.imshow(grad_xy_8u, cmap='gray'), plt.title('Sobel X+Y - CV_8U'), plt.axis('off')

# CV_32F
plt.subplot(3, 3, 4), plt.imshow(grad_x_32f_disp, cmap='gray'), plt.title('Sobel X - CV_32F'), plt.axis('off')
plt.subplot(3, 3, 5), plt.imshow(grad_y_32f_disp, cmap='gray'), plt.title('Sobel Y - CV_32F'), plt.axis('off')
plt.subplot(3, 3, 6), plt.imshow(grad_xy_32f_disp, cmap='gray'), plt.title('Sobel X+Y - CV_32F'), plt.axis('off')

# CV_64F
plt.subplot(3, 3, 7), plt.imshow(grad_x_64f_disp, cmap='gray'), plt.title('Sobel X - CV_64F'), plt.axis('off')
plt.subplot(3, 3, 8), plt.imshow(grad_y_64f_disp, cmap='gray'), plt.title('Sobel Y - CV_64F'), plt.axis('off')
plt.subplot(3, 3, 9), plt.imshow(grad_xy_64f_disp, cmap='gray'), plt.title('Sobel X+Y - CV_64F'), plt.axis('off')

plt.tight_layout()
plt.show()
