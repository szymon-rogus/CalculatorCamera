import argparse
import numpy as np
import cv2.cv2 as cv
import os
import tensorflow as tf


EYES_CASCADE = cv.CascadeClassifier('haarcascade_eye.xml')
FACE_CASCADE = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
HAND_CASCADE = cv.CascadeClassifier('handcascade.xml')
FINGER_CASCADE = cv.CascadeClassifier('fingertip_cascade.xml')
CAMERA = 0


def camera():
    cap = cv.VideoCapture(CAMERA)

    if not cap.isOpened():
        print(f"Failed to open: {CAMERA}")
        return

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 600)

    while True:
        ret, img = cap.read()
        counter = 0
        cv.rectangle(img, (0, 0), (400, 400), (255, 0, 0), 3)
        finger_square = img[0:400, 0:400]
        fingers = FINGER_CASCADE.detectMultiScale(finger_square)
        for (x, y, w, h) in fingers:
            counter += 1
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 127, 255), 2)
        cv.putText(img, 'Number: ' + str(counter), (50, 450), cv.FONT_ITALIC, 1, (0, 255, 0),
                thickness=2, lineType=cv.LINE_AA)
        cv.imshow('img', img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()

    # Closes all the frames
    cv.destroyAllWindows()


camera()
