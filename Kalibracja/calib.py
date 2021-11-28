import numpy as np
import cv2 as cv
import glob
import os
import time
import csv

CHESSBOARD_SIZE = (6, 8)
SIZE_OF_CHESSBOARD_SQUARS_MM = 28.67
FRAME_SIZE = (1280, 1024)  # could be set on runtime depending on photo size

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0],
                       0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp = objp * SIZE_OF_CHESSBOARD_SQUARS_MM

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
objpointsLeft = []  # 3d point in real world space - left cam
imgpointsLeft = []  # 2d points in image plane. - left cam
objpointsRight = []  # 3d point in real world space - right cam
imgpointsRight = []  # 2d points in image plane. -right cam
# obrazy z poprawnie wykrytą szachownicą dla prawej kamery - par wewn.
imagesRightCam = []
# obrazy z poprawnie wykrytą szachownicą dla lewej kamery - par wewn.
imagesLeftCam = []
# obrazy z poprawnie wykrytą szachownicą dla jednej z kamer to mają być pary zdjęć dla prawej i lewej - to do parametrow zewnetrznych
imagesLeftRightCam = []


def provide_date_for_calib():
    # get relative path
    # dirname = os.path.join(os.path.realpath('.'), '..', 'src','s1', '*.png')

    # images = glob.glob(dirname)
    images = glob.glob('*.png')
    for fname in images:
        print('filename: {}'.format(fname))
        # img = cv2.imread('sample_image.png', cv2.IMREAD_COLOR) could be put in try cache
        img = cv.imread(fname)
        try:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        except:
            print('Error cvtColor {}'.format(fname))
            continue
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        # If found, add object points, image points (after refining them)
        if ret:
            # for stereo calib use
            # corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            handle_add_to_list(fname, corners)
            # Draw and display the corners
            # show_img(img, gray, corners)
    # I do not like the solution, but do not have idea for better one in this moment
    global imgForCalib
    imgForCalib = img
    cv.destroyAllWindows()


def show_img(img, gray, corners):
    # for single calib use
    corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    cv.drawChessboardCorners(img, CHESSBOARD_SIZE, corners)
    cv.imshow('img', img)
    cv.waitKey(500)


def handle_add_to_list(filename, corners):
    if filename.find('left') >= 0:
        imagesLeftCam.append(filename)
        objpointsLeft.append(objp)
        imgpointsLeft.append(corners)
    elif filename.find('right') >= 0:
        imagesRightCam.append(filename)
        objpointsRight.append(objp)
        imgpointsRight.append(corners)


def create_list_img_left_right():
    floorIndex = get_number_index(imagesLeftCam[0])
    leftImagesId = [filename[floorIndex:] for filename in imagesLeftCam]
    floorIndex = get_number_index(imagesRightCam[0])
    rightImagesId = [filename[floorIndex:] for filename in imagesRightCam]

    # Adding photo numbers to list
    imagesLeftRightCam.extend(set(leftImagesId).intersection(rightImagesId))
    # Adding correct number of 3d points to list
    objpoints.extend(objpointsLeft[:len(imagesLeftRightCam)])


def get_number_index(filename):
    return filename.find('_') + 1


def save_to_csv(listToSave, fileName):
    with open("{}.csv".format(fileName), "w", encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(listToSave)


def calib_single_cam(objpointsArg, imgpointsArg):
    # Calibration
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpointsArg, imgpointsArg, FRAME_SIZE, None, None)

    height, width, channels = imgForCalib.shape
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (width, height), 1, (width, height))

    # Undistort
    dst = cv.undistort(imgForCalib, cameraMatrix, dist, None, newCameraMatrix)

    # Undistort with Remapping
    mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (width, height), 5)
    dst = cv.remap(imgForCalib, mapx, mapy, cv.INTER_LINEAR)

    # Not sure which matrix we should save
    return cameraMatrix, dst


def save_single_calib_to_xml(cameraMatrix, map, filename):
    cv_file = cv.FileStorage('{}.xml'.format(filename), cv.FILE_STORAGE_WRITE)
    cv_file.write('camera_matrix', cameraMatrix)
    cv_file.write('map_x', map[0])
    cv_file.write('map_y', map[1])
    cv_file.release()


def calib_stereo_cam(): #sort all list missing 
    # Calibration Left Cam
    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpointsLeft, imgpointsLeft, FRAME_SIZE, None, None)
    newCameraMatrixL, roiL = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, FRAME_SIZE, 1, FRAME_SIZE)

    # Calibration Right Cam
    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpointsRight, imgpointsRight, FRAME_SIZE, None, None)
    newCameraMatrixR, roiR = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, FRAME_SIZE, 1, FRAME_SIZE)

    # Stereo vision calibration
    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC
    
    criteria_stereo= (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsLeft, imgpointsRight, newCameraMatrixL, distL, newCameraMatrixR, distR, imgForCalib.shape[::-1], criteria_stereo, flags)

    # Shuld save some matrix, not sure which

    # Stereo Rectification
    
    rectifyScale= 1
    rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, imgForCalib.shape[::-1], rot, trans, rectifyScale,(0,0))

    stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, imgForCalib.shape[::-1], cv.CV_16SC2)
    stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, imgForCalib.shape[::-1], cv.CV_16SC2)

    save_stereo_config(stereoMapL, stereoMapR, "stereoConfig")

def save_stereo_config(mapL, mapR, filename):
    cv_file = cv.FileStorage('{}.xml'.format(filename), cv.FILE_STORAGE_WRITE)
    cv_file.write('stereoMapL_x', mapL[0])
    cv_file.write('stereoMapL_y', mapL[1])
    cv_file.write('stereoMapR_x', mapR[0])
    cv_file.write('stereoMapR_y', mapR[1])
    cv_file.release()


def main():
    start = time.time()
    provide_date_for_calib()

    # # Left Cam
    # cameraMatrix, dst = calib_single_cam(objpointsLeft, imgpointsLeft)
    # save_single_calib_to_xml(cameraMatrix, dst, "leftCamConfig")

    # # Right Cam
    # cameraMatrix, dst = calib_single_cam(objpointsRight, imgpointsRight)
    # save_single_calib_to_xml(cameraMatrix, dst, "rightCamConfig")

    create_list_img_left_right()
    calib_stereo_cam()
    print("Run Time = {:.2f}".format(time.time() - start))
    # print(imagesLeftCam)
    # print(imagesRightCam)
    # print(imagesLeftRightCam)
    # save_to_csv(imagesLeftRightCam, "imagesLeftRightCam")


if __name__ == "__main__":
    main()
