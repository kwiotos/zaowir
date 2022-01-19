import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math

def get_disparity_vis(src: np.ndarray, scale: float = 1.0) -> np.ndarray:
    '''Replicated OpenCV C++ function

    Found here: https://github.com/opencv/opencv_contrib/blob/b91a781cbc1285d441aa682926d93d8c23678b0b/modules/ximgproc/src/disparity_filters.cpp#L559
    
    Arguments:
        src (np.ndarray): input numpy array
        scale (float): scale factor

    Returns:
        dst (np.ndarray): scaled input array
    '''
    dst = (src * scale/16.0).astype(np.uint8)
    return dst

def disp2deep(disp, b, fov):
	h, w = disp.shape[:2]
	# print("h, w :", h, w)
	focus_length = w/(2*math.tan(fov*math.pi/360))
	# center_x = w/2
	# center_y = h/2
	Z = (b*focus_length)/disp
	return Z

def get_point3D(disp, u, v, b, fov):
	h, w = disp.shape[:2]

	focus_length = len(disp)/(2*math.tan(fov*math.pi/360))
	Z = (b*focus_length)/disp
	K = np.array([[focus_length, 0, w/2], [0, focus_length, h/2], [0, 0, 1]])

	M = Z * np.linalg.inv(K) * np.array([[u],[v],[1]])


# imgL = cv.imread('ambush_5_left.jpg')
# imgR = cv.imread('ambush_5_right.jpg')

imgL = cv.imread('334L.png')
imgR = cv.imread('334P.png')

# params for ambush
# wsize=13
# max_disp = 128
# sigma = 1.0
# lmbda = 8000.0

# params for carla
wsize= 27
max_disp = 128
sigma = 2
lmbda = 8000.0

baseline = 0.3
FOV = 120

# max_disp/=2
# if max_disp%16 != 0:
#     max_disp += 16-(max_disp%16)
# max_disp = int(max_disp)
# left_for_matcher = cv.resize(imgL,cv.Size(),0.5,0.5, INTER_LINEAR_EXACT);
# right_for_matcher = cv.resize(imgR,cv.Size(),0.5,0.5, INTER_LINEAR_EXACT);

left_for_matcher = imgL
right_for_matcher = imgR

left_matcher = cv.StereoBM_create(max_disp, wsize);
wls_filter = cv.ximgproc.createDisparityWLSFilter(left_matcher);
right_matcher = cv.ximgproc.createRightMatcher(left_matcher);

left_for_matcher = cv.cvtColor(left_for_matcher, cv.COLOR_BGR2GRAY)
right_for_matcher = cv.cvtColor(right_for_matcher, cv.COLOR_BGR2GRAY)

left_disp = left_matcher.compute(left_for_matcher, right_for_matcher);
right_disp = right_matcher.compute(right_for_matcher, left_for_matcher);

# Now create DisparityWLSFilter

wls_filter.setLambda(lmbda);
wls_filter.setSigmaColor(sigma);
filtered_disp = wls_filter.filter(left_disp, imgL, disparity_map_right=right_disp);


vis_filtered_disp = get_disparity_vis(filtered_disp,scale = 1.0)

#cv.imshow("raw disparity", get_disparity_vis(left_disp,scale = 4.0 ))
cv.imshow("filtered disparity", vis_filtered_disp)

deep_map = disp2deep(vis_filtered_disp, baseline, FOV)

#converted_deep_map = cv.cvtColor(deep_map, cv.COLOR_BGR2GRAY)

#print("deep car :", get_point3D(vis_filtered_disp,894,343, baseline, FOV))
cv.imshow("deep from disparity", deep_map)
print("point", deep_map[735,1170])
print("point", deep_map[345,892])
print("shape", deep_map.shape)

#cv.imwrite("deep_from_disparity.png", deep_map)

# plt.imshow(deep_map, 'gray')
# plt.show()

#cv.imwrite("filtered_disparity_auto_test.png", vis_filtered_disp)


print("fine")
cv.waitKey()