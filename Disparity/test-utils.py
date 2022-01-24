import cv2 as cv
from matplotlib import pyplot as plt
from utils import deep2disp, get_camera_matrix, carla2deep


pic = cv.imread('deep/334_carla_deep.png')

deep_from_carla = carla2deep(pic)

disparity = deep2disp(deep_from_carla, 0.3, 120)

cameraMatrix = get_camera_matrix(120, disparity)

print(f'Camera Matrix: {cameraMatrix}')

# plt.imshow(disparity, 'gray')
# plt.imshow(deep_from_carla, 'gray')
# plt.show()
