import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_vari(rgb_image):
    blue, green, red = cv2.split(rgb_image)
    vari = (green - red) / (green + red - blue + 0.1)
    return vari

def normalize_vari(vari_result):
    return ((vari_result + 1) * 127.5).astype(np.uint8)

def apply_threshold_colormap(vari_normalized, threshold):
    rows, cols = vari_normalized.shape[:2]
    color_map = np.zeros((rows, cols, 3), dtype=np.uint8)
    
    # Apply the threshold-based colormap 
    for i in range(rows):
        for j in range(cols):
            if vari_normalized[i, j] < threshold:
                color_map[i, j] = [255, 0, 0]  # Blue for values below threshold
            else:
                color_map[i, j] = [0, 255, 0]  # Green for values above or equal to threshold
    
    return color_map

# Read the original image
rgb_image = cv2.imread(r"images/land.jpg")

# Calculate VARI
vari_result = calculate_vari(rgb_image)

# Normalize VARI
vari_normalized = normalize_vari(vari_result)

# Define the threshold
threshold = 100

# Apply threshold-based colormap
color_map = apply_threshold_colormap(vari_normalized, threshold)
# Save the colour map image
cv2.imwrite("colour_map.jpg",color_map)
# Plot side by side using Matplotlib, Optional
plt.figure(figsize=(10, 5))

# Plot original image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Plot VARI thresholded image
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB))
plt.title('VARI Visualization - Threshold Colormap')
plt.axis('off')

plt.show()
