import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob



def nothing(x):
  pass



def webcam():
    cam = cv.VideoCapture(0)
    if cam is None or not cam.isOpened():
        print('Warning: unable to open video source')
        return 0

    cam.set(3, 1280)  # width and height
    cam.set(4, 720)

    cv.namedWindow('trackbars')
    cv.createTrackbar('threshold_low', 'trackbars', 0, 255, nothing)
    cv.createTrackbar('threshold_high', 'trackbars', 0, 255, nothing)

    while True:
        try:
            success, frame = cam.read()
        except:
            print("Cam not available.")
            break

        #  write program here

        threshold_low = cv.getTrackbarPos('threshold_low', 'trackbars')
        threshold_high = cv.getTrackbarPos('threshold_high', 'trackbars')

        canny = cv.Canny(frame, threshold_low, threshold_high)
        cv.imshow('WINDOW', canny)

   



        # end program here

        if cv.waitKey(20) & 0xFF == ord('q'):
            break

    cam.release()
    cv.destroyAllWindows()


webcam()






