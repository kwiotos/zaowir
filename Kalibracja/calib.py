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


def calib_cam():
    # get relative path
    # dirname = os.path.join(os.path.realpath('.'), '..', 'src','s1', '*.png')

    # images = glob.glob(dirname)
    images = glob.glob('*.png')
    for fname in images:
        print('filename: {}'.format(fname))
        img = cv.imread(fname)
        try:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        except:
            print('Error cvtColor {}'.format(fname))
            continue
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(img, CHESSBOARD_SIZE, None)
        # If found, add object points, image points (after refining them)
        if ret:
            handle_add_to_list(fname, corners)
            # Draw and display the corners
            # show_img(img, gray, corners)
    cv.destroyAllWindows()


def show_img(img, gray, corners):
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    cv.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2)
    cv.imshow('img', img)
    cv.waitKey(500)


def handle_add_to_list(filename, corners):
    if filename.find('left') >= 0:
        imagesLeftCam.append(filename)
        objpointsLeft.append(objp)
        imgpointsLeft.append(corners)
    elif filename.find('right') >= 0:
        imagesRightCam.append(filename)
        objpointsLeft.append(objp)
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


def save_to_csv():
    with open("output.csv", "w", encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(imagesLeftRightCam)


def main():
    start = time.time()
    calib_cam()
    create_list_img_left_right()
    print("Run Time = {:.2f}".format(time.time() - start))
    # print(imagesLeftCam)
    # print(imagesRightCam)
    print(imagesLeftRightCam)
    save_to_csv()


if __name__ == "__main__":
    main()
