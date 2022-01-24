import numpy as np
import cv2 as cv
import json
import math


ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save2json(data, filename):
    with open(filename, "w", encoding='UTF8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)


def loadFromJson(filename) -> dict:
    # Opening JSON file
    f = open(filename)

    # returns JSON object as
    # a dictionary
    data = json.load(f)
    f.close()

    return data


def disp2deep(disp, b, fov):
    _, w = disp.shape[:2]
    focus_length = w/(2*math.tan(fov*math.pi/360))
    Z = (b*focus_length)/disp
    return Z


def deep2disp(deep, b, fov):
    _, w = deep.shape[:2]
    focus_length = w/(2*math.tan(fov*math.pi/360))
    disp = (b*focus_length)/deep
    return disp.astype(np.uint8)


def carla2deep(array):
    array = array[:, :, :3]  # Take only RGB
    array = array[:, :, ::-1]  # BGR
    gray_depth = ((array[:, :, 0] + array[:, :, 1] * 256.0 +
                  array[:, :, 2]*256.0*256.0)/((256.0*256.0*256.0) - 1))
    gray_depth = gray_depth * 1000
    return gray_depth


def get_camera_matrix(fov, img):
    h, w = img.shape[:2]
    focus_length = w/(2*math.tan(fov*math.pi/360))
    K = np.array([[focus_length, 0, w/2], [0, focus_length, h/2], [0, 0, 1]])
    return K


def write_ply(fn, disparity):
    print('generating 3d point cloud...',)
    h, w = disparity.shape[:2]
    f = 0.8*w # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
    [0,-1, 0, 0.5*h], # turn points 180 deg around x-axis,
    [0, 0, 0, -f], # so that y-axis looks up
    [0, 0, 1, 0]])
    points = cv.reprojectImageTo3D(disparity, Q)
    colors = cv.cvtColor(disparity, cv.COLOR_BGR2RGB)
    mask = disparity > disparity.min()
    verts = points[mask]
    colors = colors[mask]

    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')