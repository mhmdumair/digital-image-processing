import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------- 1. Load & pre-process -----------------
img = cv2.imread("../image/square.jpg")              # <-- change filename
if img is None:
    raise IOError('Image not found!')

gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur   = cv2.GaussianBlur(gray, (7, 7), 0)
_, bin = cv2.threshold(blur, 0, 255,
                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# invert if the object is darker than background
# bin = cv2.bitwise_not(bin)

# ----------------- 2. Find contours ----------------------
contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

cnt = max(contours, key=cv2.contourArea)

# ----------------- 3. Compute & annotate perimeter --------
perimeter = cv2.arcLength(cnt, True)        # True → closed contour
annotated = img.copy()
cv2.drawContours(annotated, [cnt], -1, (255, 0, 0), 2)

text  = f'Perimeter: {perimeter:.2f} px'
cv2.putText(annotated, text, (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# ----------------- 4. Display -----------------------------
annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
plt.imshow(annotated_rgb)
plt.title('Largest Contour – Perimeter')
plt.axis('off')
plt.show()
