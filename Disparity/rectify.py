import cv2 as cv
import time
import numpy as np
from utils import loadFromJson


def scale_pic(img, scale_percent = 60):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    return cv.resize(img, dim, interpolation = cv.INTER_AREA)

def im_show(name, img, scale = 60):
    cv.imshow(name, scale_pic(img, scale))

def rectify(cameraMatrix, dist, rect, projMatrix, imgRectify, size):
    # Undistort with Remapping
    mapx, mapy = cv.initUndistortRectifyMap(
        cameraMatrix, dist, rect, projMatrix, size, cv.CV_16SC2)
    dst = cv.remap(imgRectify, mapx, mapy,
                   cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    return dst


def undistortion_with_map(cameraMatrix, dist, newCameraMatrix, imgUnditort, size):
    # h, w, _ = imgUnditort.shape 
    # # Undistort with Remapping
    # mapx, mapy = cv.initUndistortRectifyMap(
    #     cameraMatrix, dist, None, newCameraMatrix, size, 5)
    # dst = cv.remap(imgUnditort, mapx, mapy, cv.INTER_LINEAR)
    dst = cv.undistort(imgUnditort, cameraMatrix, dist, None, newCameraMatrix)
    return dst


def stereoRectify(leftPic, rightPic, size):
    # Load data from json
    data = loadFromJson("stereo_calib_config.json")

    im_show("leftPicBefore", leftPic)
    im_show("rightPicBefore", rightPic)

    leftPic = undistortion_with_map(np.array(data['cameraMatrixL']),  np.array(data['distL']), np.array(data['newCameraMatrixL']), leftPic, size)
    rightPic = undistortion_with_map(np.array(data['cameraMatrixR']),  np.array(data['distR']), np.array(data['newCameraMatrixR']), rightPic, size)

    im_show("leftPic", leftPic)
    im_show("rightPic", rightPic)

    # Stereo Rectification
    rectifyScale = 1
    rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(
        np.array(data['newCameraMatrixL']), np.array(data['distL']), np.array(data['newCameraMatrixR']), np.array(data['distR']), size, np.array(data['rot']), np.array(data['trans']), rectifyScale, (0, 0))

    leftRectified = rectify(
        np.array(data['newCameraMatrixL']), np.array(data['distL']), rectL, projMatrixL, leftPic, size)
    rightRectified = rectify(
        np.array(data['newCameraMatrixR']), np.array(data['distR']), rectR, projMatrixR, rightPic, size)

    # missing show pics
    im_show("leftRectified", leftRectified)
    im_show("rightRectified", rightRectified)
    cv.waitKey(200000)


def main():
    start = time.time()

    right = cv.imread("prawe_5.png")
    left = cv.imread("lewe_5.png")
    h, w = left.shape[:2] 

    stereoRectify(left, right, (w, h))  # missing pics

    print("Run Time = {:.2f}".format(time.time() - start))


if __name__ == "__main__":
    main()
