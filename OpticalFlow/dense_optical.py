import numpy as np
import cv2 as cv
cap = cv.VideoCapture(cv.samples.findFile("slow_traffic_small.mp4"))
# cap = cv.VideoCapture(cv.samples.findFile("parking.mp4"))
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
while(1):
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    img_gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

    ret,thresh1 = cv.threshold(img_gray,45,255,cv.THRESH_BINARY)

    kernel = np.ones((5,5),np.uint8)
    opening = cv.morphologyEx(thresh1, cv.MORPH_OPEN, kernel)

    contours, heirarchy = cv.findContours(opening, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        # Zaznaczenie obiektu o odpowiednio dużym rozmiarze
        if 60 < w < 250 and 60 < h < 250: 
            cpy = bgr.copy()
            cropped_image = cpy[int(y+0.25*h):int(y+0.75*h), int(x+0.25*w):int(x+0.75*w)]
            b,g,r = cropped_image.mean(axis=0).mean(axis=0)
            brightness = (0.2126*b + 0.7152*g + 0.0722*r)/255
            # Zaznaczenie obiektu o odpowiednio dużej jasności - prędkości
            if brightness > 0.4:
                cv.putText(frame2, "High speed", (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (b,g,r), 2, cv.LINE_AA)
            # Zaznaczenie obiektu o odpowiednio dużej składowej czerwonej średniego koloru
            if r > 130:
                cv.rectangle(frame2, (int(x), int(y)), (int(x+w), int(y+h)),(b,g,r), thickness=2)  

    cv.imshow('optical', bgr)
    cv.imshow('final', frame2)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame2)
        cv.imwrite('opticalhsv.png', bgr)
    prvs = next

cv.destroyAllWindows()