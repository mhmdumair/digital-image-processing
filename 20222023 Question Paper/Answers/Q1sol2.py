import cv2
import numpy as np
import matplotlib.pyplot as plt

def selective_blur_inside_contour(image, min_perimeter):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on perimeter
    filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) > min_perimeter]

    # Create a binary mask for the contours
    # mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # cv2.drawContours(mask, filtered_contours, -1, 255, -1)

    # Create a structuring element for morphological operations
    kernel = np.ones((15, 15), np.uint8)

    # Apply Top Hat operation to the grayscale image
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    
    # Convert the single-channel top_hat image to a three-channel image
    top_hat_bgr = cv2.cvtColor(top_hat, cv2.COLOR_GRAY2BGR)
    mask = gray-top_hat
    _, mask_th = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    # Blend Top Hat image with the original using the mask
    blurred_image = cv2.GaussianBlur(image, (43, 43), 0)
    result = image.copy()
    result[mask_th != 255] = blurred_image[mask_th != 255]

    # Visualize the process
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Selective Blurring Output')
    plt.axis('off')

    contour_image = result.copy()
    cv2.drawContours(contour_image, filtered_contours, -1, (0, 255, 0), 2)

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    plt.title('Boundary of the Galaxy')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return result

image = cv2.imread(r'images\galaxy.jpg')

# Define minimum perimeter for contours
min_perimeter = 100  # Adjust as needed

# Perform selective blurring inside the contour using Top Hat operation
blurred_image = selective_blur_inside_contour(image, min_perimeter)
