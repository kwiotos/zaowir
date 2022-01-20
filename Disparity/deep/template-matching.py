import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math

TEMPLATE_WINDOW_SIZE = 9
TEMPLATE_WINDOW_RADIUS = (TEMPLATE_WINDOW_SIZE-1)//2

def disp2deep(disp, b, fov):
	h, w = disp.shape[:2]
	focus_length = w/(2*math.tan(fov*math.pi/360))
	Z = (b*focus_length)/disp
	return Z

def findMatchingPointX(image, template):
    # methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
    #             'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    res = cv.matchTemplate(image,template, cv.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    w, h = template.shape[::-1]
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    return min_loc[0] + (w-1)/2

#teddy images
# try:
#     left = (cv.imread('imL.png', 0)/4).astype('uint8')
#     right = (cv.imread('imP.png', 0)/4).astype('uint8')
# except:
#     print("Error reading image")

try:
    leftL = cv.imread('334L.png', 0)
    left = cv.resize(leftL, (leftL.shape[1]//4, leftL.shape[0]//4), interpolation = cv.INTER_AREA)
    rightP = cv.imread('334P.png', 0)
    right = cv.resize(rightP, (rightP.shape[1]//4, rightP.shape[0]//4), interpolation = cv.INTER_AREA)
except:
    print("Error reading image")

disparity = right.copy()

wL, hL = left.shape[::-1]

for y in range(TEMPLATE_WINDOW_RADIUS, hL-1):
    if (hL - y) <= TEMPLATE_WINDOW_RADIUS:
        break
    for x in range(TEMPLATE_WINDOW_RADIUS, wL-1):
        if (wL - x) <= TEMPLATE_WINDOW_RADIUS:
            break
        disparity[y,x] = x - findMatchingPointX(
            right[(y-TEMPLATE_WINDOW_RADIUS):(y+TEMPLATE_WINDOW_RADIUS),:], 
            left[(y-TEMPLATE_WINDOW_RADIUS):(y+TEMPLATE_WINDOW_RADIUS),(x-TEMPLATE_WINDOW_RADIUS):(x+TEMPLATE_WINDOW_RADIUS)]
            )

depth = disp2deep(disparity, 0.3, 120)

f, axarr = plt.subplots(1,2)
# axarr[0].imshow(left,'gray')
# axarr[1].imshow(right,'gray')
axarr[0].imshow(disparity, 'gray')
axarr[1].imshow(depth, 'gray')
plt.show()