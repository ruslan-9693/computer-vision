import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
# import imghdr

# Example of useful code

# Convert to grayscale:
# if read_method == mpimg.imread():
#     gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
# elif read_method == cv2.imread():
#     gray = cv2.COLOR_BGR2GRAY

# Calculate derivative in x direction:
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)

# Calculate derivative in y direction:
#     sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

# Calculate absolute value of x derivative:
#     abs_sobelx = np.absolute(sobelx)

# Convert absolute value image to 8-bit:
#     scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

# Create binary threshold to select pixels based on gradient strength:
#     thresh_min = 20
#     thresh_max = 100
#     sxbinary = np.zeros_like(scaled_sobel)
#     sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
#     plt.imshow(sxbinary, cmap='gray')

image = mpimg.imread('signs_vehicles_xygrad.png')


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, int(orient == 'x'), int(orient == 'y'))
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return mag_binary


def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary = np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return dir_binary


def plot(original_image, processed_image, processed_image_title):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(original_image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(processed_image, cmap='gray')
    ax2.set_title(processed_image_title, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


# Run the function
grad_binary = abs_sobel_thresh(image, orient='x', thresh=(20, 100))
# Plot the result
plot(image, grad_binary, 'Thresholded Gradient')


# Run the function
mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(30, 100))
# Plot the result
plot(image, mag_binary, 'Thresholded Magnitude')


# Run the function
dir_binary = dir_thresh(image, sobel_kernel=15, thresh=(0.75, 1.25))
# Plot the result
plot(image, dir_binary, 'Thresholded Grad. Dir.')
