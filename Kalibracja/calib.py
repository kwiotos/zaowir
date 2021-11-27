import numpy as np
import cv2 as cv
import glob
import os

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
#obrazy z poprawnie wykrytą szachownicą dla prawej kamery - par wewn.
imagesRightCam = []
#obrazy z poprawnie wykrytą szachownicą dla lewej kamery - par wewn.
imagesLeftCam = []
#obrazy z poprawnie wykrytą szachownicą dla jednej z kamer to mają być pary zdjęć dla prawej i lewej - to do parametrow zewnetrznych
imagesLeftRightCam = []

# get relative path
dirname = os.path.join(os.path.realpath('.'), 'src','s1', '*.png')

images = glob.glob(dirname)
for fname in images:
    print('filename: {}'.format(fname))
    try:
        img = cv.imread(fname)
    except:
        print('Error reading image {}'.format(fname))
        continue
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (6, 8), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (6, 8), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
cv.destroyAllWindows()
