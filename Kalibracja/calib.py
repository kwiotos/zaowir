import numpy as np
import cv2 as cv
import glob
import os
import time
import csv
from tqdm import tqdm
from numpy import linalg as LA

CHESSBOARD_SIZE = (8, 6)
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
objpointsLeft = []  # 3d point in real world space - left cam
objpointsRight = []  # 3d point in real world space - right cam

imageLeft_dict = {}
imageRight_dict = {}
common_imageLeft_dict = {}
common_imageRight_dict = {}


def provide_data_for_calib():
    # get relative path
    # dirname = os.path.join(os.path.realpath('.'), '..', 'src','s1', '*.png')
    dirname = os.path.join(os.path.realpath('.'), '..', 'src', 's4', '*.png')
    #print(dirname)

    images = glob.glob(dirname)
    # images = glob.glob('*.png')
    for fname in tqdm(images):
        #print('filename: {}'.format(fname))
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
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            handle_add_to_list(fname, corners)
            # Draw and display the corners
            # show_img(img, corners, fname)
    # I do not like the solution, but do not have idea for better one in this moment
    global imgForCalib
    imgForCalib = gray
    cv.destroyAllWindows()


def show_img(img, corners, filename):
    cv.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, True)
    cv.imshow("img", img)
    cv.waitKey(2000)


def handle_add_to_list(filename, corners):
    if filename.find('left') >= 0:
        objpointsLeft.append(objp)
        floorIndex = get_number_index(filename)
        imageLeft_dict[filename[floorIndex:]] = corners

    elif filename.find('right') >= 0:
        objpointsRight.append(objp)
        floorIndex = get_number_index(filename)
        imageRight_dict[filename[floorIndex:]] = corners


def create_list_img_left_right():
    #wyszukanie wspólnych kluczy - numerów zdjęć z odnalezionymi wierzchołkami dla obu kamer
    common_keys = set(imageLeft_dict.keys()).intersection(imageRight_dict.keys())

    #zapisanie wierzchołków tablicy do odpowiedniego słownika, ale tylko dla wspólnych kluczy
    common_imageLeft_dict.update({key : imageLeft_dict[key] for key in common_keys})
    common_imageRight_dict.update({key : imageRight_dict[key] for key in common_keys})
    
    # Adding correct number of 3d points to list
    objpoints.extend(objpointsLeft[:len(common_keys)])


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

def distortion_with_map(cameraMatrix, dist, newCameraMatrix, size):
    # Undistort with Remapping
    mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, size, 5)
    dst = cv.remap(imgForCalib, mapx, mapy, cv.INTER_LINEAR)
    return dst

def save_single_calib_to_xml(cameraMatrix, map, filename):
    cv_file = cv.FileStorage('{}.xml'.format(filename), cv.FILE_STORAGE_WRITE)
    cv_file.write('camera_matrix', cameraMatrix)
    cv_file.write('map_x', map[0])
    cv_file.write('map_y', map[1])
    cv_file.release()


def calib_stereo_cam(): #sort all list missing 
    # Calibration Left Cam
    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpointsLeft, list(imageLeft_dict.values()), FRAME_SIZE, None, None)
    newCameraMatrixL, roiL = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, FRAME_SIZE, 1, FRAME_SIZE)
    dstMap = distortion_with_map(cameraMatrixL, distL, newCameraMatrixL, FRAME_SIZE)
    save_single_calib_to_xml(newCameraMatrixL, dstMap, "leftCamConfig")
    mean_error(objpointsLeft, list(imageLeft_dict.values()), rvecsL, tvecsL,cameraMatrixL, distL)

    # Calibration Right Cam
    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpointsRight, list(imageRight_dict.values()), FRAME_SIZE, None, None)
    newCameraMatrixR, roiR = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, FRAME_SIZE, 1, FRAME_SIZE)
    dstMap = distortion_with_map(cameraMatrixR, distR, newCameraMatrixR, FRAME_SIZE)
    save_single_calib_to_xml(newCameraMatrixR, dstMap, "rightCamConfig")
    mean_error(objpointsRight, list(imageRight_dict.values()), rvecsR, tvecsR, cameraMatrixR, distR)

    # Stereo vision calibration
    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC
    
    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, list(common_imageLeft_dict.values()), list(common_imageRight_dict.values()), newCameraMatrixL, distL, newCameraMatrixR, distR, imgForCalib.shape[::-1], criteria= criteria_stereo, flags= flags)

    print("Baseline: {}".format(LA.norm(trans)))
    # print("Baseline: {}".format(LA.norm(LA.inv(rot)*trans)))

    # Shuld save some matrix, not sure which

    # zapisywac wszystko, rot, trans wszystko wyzej, kod ponizej jest na kolejne labki, nie na teraz

    # Stereo Rectification
    
    # rectifyScale= 1
    # rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, imgForCalib.shape[::-1], rot, trans, rectifyScale,(0,0))

    # stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, imgForCalib.shape[::-1], cv.CV_16SC2)
    # stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, imgForCalib.shape[::-1], cv.CV_16SC2)

    save_stereo_config(retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix, "stereoConfig")


def save_stereo_config(retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix,filename):
    cv_file = cv.FileStorage('{}.xml'.format(filename), cv.FILE_STORAGE_WRITE)
    cv_file.write('retStereo', retStereo)
    cv_file.write('newCameraMatrixL', newCameraMatrixL)
    cv_file.write('distL', distL)
    cv_file.write('newCameraMatrixR',newCameraMatrixR)
    cv_file.write('distR', distR)
    cv_file.write('rot', rot)
    cv_file.write('trans', trans)
    cv_file.write('essentialMatrix', essentialMatrix)
    cv_file.write('fundamentalMatrix', fundamentalMatrix)
    cv_file.release()


def mean_error(objpointsArg, imgpointsArg, rvecs, tvecs, mtx, dist):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpointsArg[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpointsArg[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print("Mean reprojection error: {}", mean_error/len(objpoints))


def main():
    start = time.time()
    provide_data_for_calib()

    # Left Cam
    # cameraMatrix, dst = calib_single_cam(objpointsLeft, imageLeft_dict.values())
    # save_single_calib_to_xml(cameraMatrix, dst, "leftCamConfig")
    # print("Macierz wewnętrzna lewa kamera: {}\n".format(cameraMatrix))

    # Right Cam
    # cameraMatrix, dst = calib_single_cam(objpointsRight, imageRight_dict.values())
    # save_single_calib_to_xml(cameraMatrix, dst, "rightCamConfig")
    # print("Macierz wewnętrzna prawa kamera: {}\n".format(cameraMatrix))

    create_list_img_left_right()
    calib_stereo_cam()
    print("Run Time = {:.2f}".format(time.time() - start))


if __name__ == "__main__":
    main()
