import numpy as np
import cv2
import matplotlib.pyplot as plt

COLOR = 'maroon'
FONT_SIZE = 8
KERNEL_SIZE = (21,21)


img = cv2.imread('images/harbour.jpg',0)

ft_img = np.fft.fft2(img)
fshift_img = np.fft.fftshift(ft_img)
ft_shift_img_abs = np.abs(fshift_img)

# log transform for better visibility
magnitude_spectrum_shifted = 20*np.log(np.abs(fshift_img))

# band reject - a mask containing 4 black circles
rows, cols = img.shape
mask = np.ones((rows, cols), np.uint8)
r = 4

#harbour
center_1,center_2,center_3,center_4 = [150,122],[94,55],[88,21],[75,84]

#vertical
start_point_1, end_point_1 = (150,0),(150,rows) #x,y
start_point_2, end_point_2 = (167,0),(167,rows) #x,y

#horizontal
start_point_3, end_point_3 = (0,121),(cols,121) #x,y
start_point_4, end_point_4 = (0,137),(cols,137) #x,y

#diagonal
#top-left
start_point_5, end_point_5 = (15,43),(149,125) #x,y
#bottom-left
start_point_6, end_point_6 = (24,168),(147,131) #x,y
#bottom-right
start_point_7, end_point_7 = (177,141),(300,213) #x,y
#top-right
start_point_8, end_point_8 = (176,126),(304,88) #x,y

##
cv2.line(mask, start_point_1, end_point_1, (0,0,0), r)
cv2.line(mask, start_point_2, end_point_2, (0,0,0), r)
cv2.line(mask, start_point_3, end_point_3, (0,0,0), r)
cv2.line(mask, start_point_4, end_point_4, (0,0,0), r)


cv2.line(mask, start_point_5, end_point_5, (0,0,0), r)
cv2.line(mask, start_point_6, end_point_6, (0,0,0), r)
cv2.line(mask, start_point_7, end_point_7, (0,0,0), r)
cv2.line(mask, start_point_8, end_point_8, (0,0,0), r)

fshift_and_mask = fshift_img*mask
ft_shift_with_mask = np.abs(fshift_and_mask)

f_ishift = np.fft.ifftshift(fshift_and_mask)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

#mask and magnitude spectrum
plt.style.use('grayscale')
plt.figure().patch.set_facecolor('white')

plt.subplot(131)
plt.title('FT Magnitude Spectrum\nShifted to centre',c=COLOR,fontsize=FONT_SIZE)
plt.axis('off')
plt.imshow(magnitude_spectrum_shifted)

plt.subplot(132)
plt.title('Mask',c=COLOR,fontsize=FONT_SIZE)
plt.axis('off')
plt.imshow(mask)


plt.style.use('grayscale')
plt.figure().patch.set_facecolor('white')

plt.subplot(411)
plt.title('Gray Image',c=COLOR,fontsize=FONT_SIZE)
plt.axis('off')
plt.imshow(img)

plt.subplot(412)
plt.title('FT Magnitude Spectrum\nShifted to centre',c=COLOR,fontsize=FONT_SIZE)
plt.axis('off')
plt.imshow(magnitude_spectrum_shifted)

plt.subplot(413)
plt.title('mask',c=COLOR,fontsize=FONT_SIZE)
plt.axis('off')
plt.imshow(mask[:,:])
##
##
plt.subplot(414)
plt.title('Image back to spatial domain',c=COLOR,fontsize=FONT_SIZE)
plt.axis('off')
plt.imshow(img_back,cmap = 'gray')

plt.show()

plt.style.use('grayscale')
plt.figure().patch.set_facecolor('white')

plt.subplot(121)
plt.title('original image',c=COLOR,fontsize=FONT_SIZE)
plt.axis('off')
plt.imshow(img)

plt.subplot(122)
plt.title('img - patterned noise removed',c=COLOR,fontsize=FONT_SIZE)
plt.axis('off')
plt.imshow(img_back)
plt.show()
