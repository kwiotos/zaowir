import numpy as np
import cv2 as cv
from utils import carla2deep, deep2disp


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
    

def main():
    pic = cv.imread('deep/334_carla_deep.png')
    deep_from_carla = carla2deep(pic)
    disparity = deep2disp(deep_from_carla, 0.3, 120) 
    # disparity = np.float32(np.divide(disparity, 16.0))
    out_fn = 'out.ply'
    write_ply(out_fn, disparity)
    print('%s saved' % out_fn)
    print('Done')


if __name__ == '__main__':
    main()
