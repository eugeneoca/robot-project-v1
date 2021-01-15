import numpy as np
import cv2 as cv
from parts.error import *
from threading import Thread
from time import sleep

class Vision():

    def __init__(self):
        self.error = Error() # Error helper
        self.cam = cv.VideoCapture(0)
        self.ml_input = None
        self.die = False
        if not self.cam.isOpened():
            self.error.report("Cannot open camera", Error.high)

    def process(self):
        while not self.die:
            ret, frame = self.cam.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            cv.imshow("Vision", frame)
            if cv.waitKey(1) == ord('q'):
                break
            self.ml_input = self.translate_input(frame)

    def translate_input(self, frame):
        resized_image = cv.resize(frame, (400,300))
        grayscale_image = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
        _, binarized_image = cv.threshold(grayscale_image, 127, 255, cv.THRESH_BINARY)
        return binarized_image.flatten()

    def get_input(self):
        return self.ml_input

    def start(self):
        v_th = Thread(target=self.process, name="Vision")
        v_th.daemon = True
        v_th.start()
        sleep(1)

    def stop(self):
        self.die = True