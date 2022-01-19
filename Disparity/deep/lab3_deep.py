import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
'''
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
'''
imgL = cv.imread('334_left.png',0)
imgR = cv.imread('334_right.png',0)

stereo = cv.StereoSGBM_create(numDisparities=80, blockSize=9)
disparity = stereo.compute(imgL,imgR)
h, w = imgL.shape[:2]
plt.imsave('disparity_SZARE.png', disparity, cmap = 'gray')
disparity2 = cv.imread('disparity_SZARE.png')
Focus_length = len(disparity2)/(2math.tan(120math.pi/360))
center_x = w/2
center_y = h/2
Z = (0.3*Focus_length)/disparity2

plt.imshow(Z, 'gray')
plt.show()
plt.imshow(disparity2, 'gray')
plt.show()