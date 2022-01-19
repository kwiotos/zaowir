import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math


def carla2deep(array):
	#array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
	#array = np.reshape(array, (image.height, image.width, 4)) # RGBA format
	array = array[:, :, :3] # Take only RGB
	array = array[:, :, ::-1] # BGR
	#array = array.astype(np.float32) # 2ms
	gray_depth = ((array[:,:,0] + array[:,:,1] *256.0 + array[:,:,2]*256.0*256.0)/((256.0*256.0*256.0) - 1)) # 2.5ms
	gray_depth = gray_depth * 1000
	return gray_depth 


carla_img = cv.imread('334_carla_deep.png', flags=cv.IMREAD_COLOR)
#cv.imshow("carla color", carla_img)

carla_converted = carla2deep(carla_img)
print("point", carla_converted[735,1170])
print("point", carla_converted[345,892])
print("shape", carla_converted.shape)
# norm = np.linalg.norm(carla_converted)

plt.imshow(carla_converted, 'gray')


cv.imshow("carla depth", carla_converted)
cv.imwrite("carla_depth.png", carla_converted)

plt.show()
cv.waitKey()