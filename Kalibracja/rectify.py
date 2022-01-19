import cv2 as cv
import time
from utils import save2json, loadFromJson


def rectify(cameraMatrix, dist, rect, projMatrix, imgRectify):
    # Undistort with Remapping
    mapx, mapy = cv.initUndistortRectifyMap(
        cameraMatrix, dist, rect, projMatrix, imgRectify.shape[::-1], cv.CV_16SC2)
    dst = cv.remap(imgRectify, mapx, mapy, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    return dst


def stereoRectify(leftPic, rightPic):
    # Load data from json
    data = loadFromJson("stereo_calib_config.json")

    # Stereo Rectification
    rectifyScale= 1
    rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(data['newCameraMatrixL'], data['distL'], data['newCameraMatrixR'], data['distR'], imgForCalib.shape[::-1], data['rot'], data['trans'], rectifyScale,(0,0))

    leftCamPhoto = rectify(data['newCameraMatrixL'], data['distL'], rectL, projMatrixL, imgRectify=leftPic)
    leftCamPhoto = rectify(data['newCameraMatrixR'], data['distR'], rectR, projMatrixR, imgRectify=rightPic)

    # missing show pics

def main():
    start = time.time()

    stereoRectify(None, None) # missing pics
    
    print("Run Time = {:.2f}".format(time.time() - start))


if __name__ == "__main__":
    main()