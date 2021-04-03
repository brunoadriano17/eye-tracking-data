"""Webcam data source for gaze estimation."""
import cv2 as cv
import numpy as np

from .frames import FramesSource


class Webcam(FramesSource):
    """Webcam frame grabbing and preprocessing."""

    def __init__(self, camera_id=0, fps=60, calibracao=None, **kwargs):
        """Create queues and threads to read and preprocess data."""
        self._short_name = 'Webcam'

        self._capture = cv.VideoCapture(camera_id)
        self._capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self._capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

        # self._capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        # self._capture.set(cv.CAP_PROP_FRAME_HEIGHT, 360)

        self._capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
        self._capture.set(cv.CAP_PROP_FPS, fps)

        if calibracao is None:
            self._calibracao = []
        else:
            self._calibracao = calibracao
        # Call parent class constructor
        super().__init__(**kwargs)

    def frame_generator(self):
        """Read frame from webcam."""
        if not self._calibracao:
            while True:
                ret, bgr = self._capture.read()
                if ret:
                    yield bgr
        else:
            for x in self._calibracao:
                yield x
