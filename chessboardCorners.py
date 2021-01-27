import cv2
import matplotlib.pyplot as plt

nx = 8
ny = 6

fname = 'calibration_test.png'
img = cv2.imread(fname)  # reads in BGR image

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

retvals, corners = cv2.findChessboardCorners(gray, (nx, ny))

if retvals is True:
    cv2.drawChessboardCorners(img, (nx, ny), corners, retvals)
    plt.imshow(img)
    plt.show()
