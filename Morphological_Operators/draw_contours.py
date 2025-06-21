import cv2
import matplotlib.pyplot as plt

# Read the image in color
img = cv2.imread("../image/plus.jpg", 1)

# Make multiple copies for different drawings
img1 = img.copy()   # for all contours
img2 = img.copy()   # for 4th contour
img3 = img.copy()   # for 6th and 7th contours

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply binary threshold
ret, thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw all contours in green
cv2.drawContours(img1, contours, -1, (0, 255, 0), 2)

# Draw 4th contour (index 3) in blue
cv2.drawContours(img2, contours, 3, (255, 0, 0), 2)

# Draw 6th and 7th contours (index 5 and 6) in red
cnt  = contours[5]
cnt1 = contours[6]
cv2.drawContours(img3, [cnt], 0, (0, 0, 255), 2)
cv2.drawContours(img3, [cnt1], -1, (0, 0, 255), 2)

img_rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img1_rgb     = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb     = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img3_rgb     = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

# Display using matplotlib
titles = ['Original Image', 'All Contours', '4th Contour', '6th and 7th contours']
images = [img_rgb, img1_rgb, img2_rgb, img3_rgb]

plt.figure(figsize=(10, 6))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
