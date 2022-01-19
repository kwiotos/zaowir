import numpy as np
import cv2 as cv
import glob
import os
import time
from tqdm import tqdm
from numpy import linalg as LA
from utils import save2json

CHESSBOARD_SIZE = (8, 6)
SIZE_OF_CHESSBOARD_SQUARS_MM = 28.67
FRAME_SIZE = (1280, 1024)

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
    dirname = os.path.join(os.path.realpath('.'), 'src', 's6', '*.png')

    images = glob.glob(dirname)
    for fname in tqdm(images):
        # print('filename: {}'.format(fname))
        img = cv.imread(fname)
        try:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        except:
            print('Error cvtColor {}'.format(fname))
            continue
        # Find the chess board corners
        # ret, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, cv.CALIB_CB_NORMALIZE_IMAGE)
        ret, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        # If found, add object points, image points (after refining them)
        if ret:
            # for stereo calib use
            corners = cv.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            handle_add_to_list(fname, corners)
            # Draw and display the corners
            # show_img(img, corners)
    global imgForCalib
    imgForCalib = gray
    cv.destroyAllWindows()


def show_img(img, corners):
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
    # wyszukanie wspólnych kluczy - numerów zdjęć z odnalezionymi wierzchołkami dla obu kamer
    common_keys = set(imageLeft_dict.keys()).intersection(
        imageRight_dict.keys())

    # zapisanie wierzchołków tablicy do odpowiedniego słownika, ale tylko dla wspólnych kluczy
    common_imageLeft_dict.update(
        {key: imageLeft_dict[key] for key in common_keys})
    common_imageRight_dict.update(
        {key: imageRight_dict[key] for key in common_keys})

    # Adding correct number of 3d points to list
    objpoints.extend(objpointsLeft[:len(common_keys)])


def get_number_index(filename):
    return filename.find('_') + 1


def undistortion_with_map(cameraMatrix, dist, newCameraMatrix, size):
    # Undistort with Remapping
    mapx, mapy = cv.initUndistortRectifyMap(
        cameraMatrix, dist, None, newCameraMatrix, size, 5)
    dst = cv.remap(imgForCalib, mapx, mapy, cv.INTER_LINEAR)
    return dst


def crop_and_save(img, roi, fname):
    x, y, w, h = roi
    img = img[y:y+h, x:x+w]
    cv.imwrite('{}.png'.format(fname), img)


def calib_stereo_cam():
    # Calibration Left Cam
    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(
        objpoints, list(common_imageLeft_dict.values()), FRAME_SIZE, None, None)
    newCameraMatrixL, roiL = cv.getOptimalNewCameraMatrix(
        cameraMatrixL, distL, FRAME_SIZE, 1, FRAME_SIZE)
    save2json({"cameraMatrixL": cameraMatrixL, "distL": distL,
              "rvecsL": rvecsL, "tvecsL": tvecsL}, "left_config.json")
    dstMap = undistortion_with_map(
        cameraMatrixL, distL, newCameraMatrixL, FRAME_SIZE)
    crop_and_save(dstMap, roiL, 'undistortedL')
    mean_error(objpoints, list(common_imageLeft_dict.values()),
               rvecsL, tvecsL, cameraMatrixL, distL)

    # Calibration Right Cam
    retL, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(
        objpoints, list(common_imageRight_dict.values()), FRAME_SIZE, None, None)
    newCameraMatrixR, roiR = cv.getOptimalNewCameraMatrix(
        cameraMatrixR, distR, FRAME_SIZE, 1, FRAME_SIZE)
    save2json({"cameraMatrixR": cameraMatrixR, "distR": distR,
              "rvecsR": rvecsR, "tvecsR": tvecsR}, "right_config.json")
    dstMap = undistortion_with_map(
        cameraMatrixR, distR, newCameraMatrixR, FRAME_SIZE)
    crop_and_save(dstMap, roiR, 'undistortedR')
    mean_error(objpoints, list(common_imageRight_dict.values()),
               rvecsR, tvecsR, cameraMatrixR, distR)

    # Stereo vision calibration
    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC

    criteria_stereo = (cv.TERM_CRITERIA_EPS +
                       cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(
        objpoints, list(common_imageLeft_dict.values()), list(
            common_imageRight_dict.values()), newCameraMatrixL, distL, newCameraMatrixR,
        distR, imgForCalib.shape[::-1], criteria=criteria_stereo, flags=flags)

    print("Baseline: {}".format(LA.norm(trans)))
    save2json({"Baseline": "{:.4f} [mm]".format(
        LA.norm(trans))}, "baseline.json")
    # print("Baseline: {}".format(LA.norm(LA.inv(rot)*trans)))

    # Shuld save some matrix, not sure which

    # zapisywac wszystko, rot, trans wszystko wyzej, kod ponizej jest na kolejne labki, nie na teraz

    # Stereo Rectification

    # rectifyScale= 1
    # rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, imgForCalib.shape[::-1], rot, trans, rectifyScale,(0,0))

    # stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, imgForCalib.shape[::-1], cv.CV_16SC2)
    # stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, imgForCalib.shape[::-1], cv.CV_16SC2)

    save2json({'retStereo': retStereo, 'newCameraMatrixL': newCameraMatrixL, 'distL': distL, 'newCameraMatrixR': newCameraMatrixR, 'distR': distR,
              'rot': rot, 'trans': trans, 'essentialMatrix': essentialMatrix, 'fundamentalMatrix': fundamentalMatrix}, "stereo_config.json")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save2json(data, filename):
    with open(filename, "w", encoding='UTF8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)


def mean_error(objpointsArg, imgpointsArg, rvecs, tvecs, mtx, dist):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(
            objpointsArg[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpointsArg[i], imgpoints2,
                        cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print("Mean reprojection error: {}".format(mean_error/len(objpoints)))


def main():
    start = time.time()
    provide_data_for_calib()

    create_list_img_left_right()

    save2json({"leftPhotoList": list(imageLeft_dict.keys()), "rightPhotoList": list(
        imageRight_dict.keys()), "commonPhotos:": list(common_imageLeft_dict.keys())}, "photo_list.json")

    calib_stereo_cam()
    print("Run Time = {:.2f}".format(time.time() - start))


if __name__ == "__main__":
    main()
