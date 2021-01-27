import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = mpimg.imread('test.png')
print('This image is: ', type(image),
      'with dimensions: ', image.shape)

ysize = image.shape[0]
xsize = image.shape[1]

color_select = np.copy(image)

red_threshold = 220/255
green_threshold = 220/255
blue_threshold = 220/255
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

thresholds = (image[:, :, 0] < rgb_threshold[0]) \
    | (image[:, :, 1] < rgb_threshold[1]) \
    | (image[:, :, 2] < rgb_threshold[2])
color_select[thresholds] = [0, 0, 0, 1]

plt.imshow(color_select)
plt.show()
