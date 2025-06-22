import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../image/telephone.jpg", cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection
minThreshold = 80
maxThreshold = 180
edges = cv2.Canny(img, minThreshold, maxThreshold)

# Display the original and edge-detected images side by side
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title(f'Canny Edges (T1={minThreshold}, T2={maxThreshold})')
plt.axis('off')

plt.tight_layout()
plt.show()
