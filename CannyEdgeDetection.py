import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

image = mpimg.imread('exit-ramp.jpg')  # reads in RGB image

# Cite: https://stackoverflow.com/questions/46689428/convert-np-array-of-type-float64-to-type-uint8-scaling-values
image = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

plt.imshow(image)
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap='gray')
plt.show()

kernel_size = 3
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

# Cite: colorSelection.py
# ! print('The image is: ', type(blur_gray),
# !        'with dimensions: ', blur_gray.shape)

low_threshold = int(1/10*255)
high_threshold = int(8/10*255)
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

plt.imshow(edges, cmap='Greys_r')
plt.show()
