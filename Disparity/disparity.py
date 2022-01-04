import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
imgL = cv.imread('im2.png', 0)/4
imgR = cv.imread('im6.png', 0)/4
stereo = cv.StereoBM_create(numDisparities=64, blockSize=5)
disparity = stereo.compute(imgL.astype('uint8'), imgR.astype('uint8'))
plt.imshow(disparity, 'gray')
plt.show()
