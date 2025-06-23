import cv2
import numpy as np
import matplotlib.pyplot as plt

def line_mask(shape):
   
    rows, cols = shape
    mask = np.ones((rows, cols), np.uint8)
    
    # Calculate the center of the mask
    center = (cols // 2, rows // 2)
    
    # Calculate the end points of the line
    thickness = 15
  
    x_start=0
    y_start=310
    x_finish=280
    y_finish=310
    # Draw the line on the mask
    cv2.line(mask, (x_start, y_start), (x_finish, y_finish), 0, thickness)

    x_start=320
    y_start=310
    x_finish=cols
    y_finish=310
    # Draw the line on the mask
    cv2.line(mask, (x_start, y_start), (x_finish, y_finish), 0, thickness)

    x_start=300
    y_start=0
    x_finish=300
    y_finish=290
    # Draw the line on the mask
    cv2.line(mask, (x_start, y_start), (x_finish, y_finish), 0, thickness)

    x_start=300
    y_start=rows-290
    x_finish=300
    y_finish=rows
    # Draw the line on the mask
    cv2.line(mask, (x_start, y_start), (x_finish, y_finish), 0, thickness)

    radius = 15
    x = 215
    y = 222
    cv2.circle(mask, (x, y), radius, 0, -1)
    
    x = 391
    y = 222
    cv2.circle(mask, (x, y), radius, 0, -1)
    
    x = 215
    y = 398
    cv2.circle(mask, (x, y), radius, 0, -1)

    x = 391
    y = 398
    cv2.circle(mask, (x, y), radius, 0, -1)







    return mask

def apply_line_mask(image):
    # Compute the 2D Fourier transform of the image
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # Create the line mask
    mask = line_mask(image.shape)
    
    # Apply the mask to the Fourier transformed image
    f_transform_filtered = f_transform_shifted * mask
    
    # Compute the inverse Fourier transform to get the filtered image
    f_transform_inverse_shifted = np.fft.ifftshift(f_transform_filtered)
    filtered_image = np.abs(np.fft.ifft2(f_transform_inverse_shifted))
    
    return filtered_image, mask

def visualize_results(image, filtered_image, mask):
    plt.figure(figsize=(18, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Compute the magnitude spectrum
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))

    plt.subplot(2, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum (zero frequency at the centre)')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap='gray')
    plt.title('Frequency Domain Mask')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Filtered Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Load the image
image = cv2.imread("../planet_surface.png", cv2.IMREAD_GRAYSCALE)

# Apply the line mask
  # Angle of the line in degrees
thickness = 10  # Thickness of the line
filtered_image, mask = apply_line_mask(image)

# Visualize the results
visualize_results(image, filtered_image, mask)
