import cv2 as cv
import time
import numpy as np
from utils import save2json, loadFromJson


def rectify(cameraMatrix, dist, rect, projMatrix, imgRectify, size):
    # Undistort with Remapping
    mapx, mapy = cv.initUndistortRectifyMap(
        cameraMatrix, dist, rect, projMatrix, size, cv.CV_16SC2)
    dst = cv.remap(imgRectify, mapx, mapy,
                   cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    return dst


def stereoRectify(leftPic, rightPic):
    # Load data from json
    data = loadFromJson("stereo_calib_config.json")


    try:
        grayL = cv.cvtColor(leftPic, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(rightPic, cv.COLOR_BGR2GRAY)
    except:
        print('Error cvtColor')
    # Stereo Rectification
    rectifyScale = 1
    rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(
        np.array(data['newCameraMatrixL']), np.array(data['distL']), np.array(data['newCameraMatrixR']), np.array(data['distR']), grayL.shape[::-1], np.array(data['rot']), np.array(data['trans']), rectifyScale, (0, 0))

    leftRectified = rectify(
        np.array(data['newCameraMatrixL']), np.array(data['distL']), rectL, projMatrixL, imgRectify=leftPic, size = grayL.shape[::-1])
    rightRectified = rectify(
        np.array(data['newCameraMatrixR']), np.array(data['distR']), rectR, projMatrixR, imgRectify=rightPic, size = grayR.shape[::-1])

    # missing show pics
    cv.imshow("leftRectified", leftRectified)
    cv.imshow("rightRectified", rightRectified)
    cv.waitKey(200000)


def main():
    start = time.time()

    right = cv.imread("prawe_5.png")
    left = cv.imread("lewe_5.png")

    stereoRectify(left, right)  # missing pics

    print("Run Time = {:.2f}".format(time.time() - start))


if __name__ == "__main__":
    main()
