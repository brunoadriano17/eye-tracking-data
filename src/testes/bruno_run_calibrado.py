import argparse
import os
import queue
import threading
import time
import win32api, win32con
import coloredlogs
import cv2 as cv
import numpy as np
import tensorflow as tf

from datasources import Video, Webcam
from models import ELG
import util.gaze

import math
import IAMLTools
import imutils
import BlobProperties
import ctypes
import time
import os
import matplotlib.pyplot as plt
import threading


if __name__ == '__main__':

    # args
    parser = argparse.ArgumentParser(description='Demonstration of landmarks localization.')
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    parser.add_argument('--from_video', type=str, help='Use this video path instead of webcam')
    parser.add_argument('--record_video', type=str, help='Output path of video of demonstration.')
    parser.add_argument('--fullscreen', action='store_true')
    parser.add_argument('--headless', action='store_true')

    parser.add_argument('--fps', type=int, default=30, help='Desired sampling rate of webcam')
    parser.add_argument('--camera_id', type=int, default=0, help='ID of webcam to use')

    args = parser.parse_args()
    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    # Verifica disponibilidade de GPU
    from tensorflow.python.client import device_lib
    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    gpu_available = False
    try:
        gpus = [d for d in device_lib.list_local_devices(config=session_config)
                if d.device_type == 'GPU']
        gpu_available = len(gpus) > 0
    except:
        pass

    # Inicia sessão
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Session(config=session_config) as session:

        # Parametros
        batch_size = 2

        # Carrega webcam para classe
        data_source = Webcam(tensorflow_session=session, batch_size=batch_size,
                                 camera_id=args.camera_id, fps=args.fps,
                                 data_format='NCHW' if gpu_available else 'NHWC',
                                 eye_image_shape=(36, 60))
        # Modelo 
        model = ELG(
            session, train_data={'videostream': data_source},
            first_layer_stride=1,
            num_modules=2,
            num_feature_maps=32,
            learning_schedule=[
                {
                    'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']},
                },
             ],
          )
        
        def _get_landmarks(output):
            last_frame_index = 0
            last_frame_time = time.time()
            fps_history = []
            all_gaze_histories = []
            count = 0;
            while True:
                # Pega saida da rede neural e exibe
                bgr = None
                for j in range(batch_size):
                    frame_index = output['frame_index'][j]
                    if frame_index not in data_source._frames:
                        continue
                    frame = data_source._frames[frame_index]
                    eye_index = output['eye_index'][j]
                    bgr = frame['bgr']
                    eye = frame['eyes'][eye_index]
                    eye_image = eye['image']
                    eye_side = eye['side']
                    eye_landmarks = output['landmarks'][j, :]
                    eye_radius = output['radius'][j][0]
                    if eye_side == 'left':
                        eye_landmarks[:, 0] = eye_image.shape[1] - eye_landmarks[:, 0]
                        eye_image = np.fliplr(eye_image)
                    # Exibir resultados pré processados
                    frame_landmarks = (frame['smoothed_landmarks']
                                       if 'smoothed_landmarks' in frame
                                       else frame['landmarks'])
                    #retorna landmarks das laterais dos olhos esquerdo e direito até [f][:-1]
                    left_eye_side_points = []
                    right_eye_side_points = []
                    for f, face in enumerate(frame['faces']):
                        for landmark in frame_landmarks[f][2:-1]:
                            left_eye_side_points.append(landmark)
                    for f, face in enumerate(frame['faces']):
                        for landmark in frame_landmarks[f][:-3]:
                            right_eye_side_points.append(landmark)
                            
                    # landmarks
                    eye_landmarks = np.concatenate([eye_landmarks,
                                                    [[eye_landmarks[-1, 0] + eye_radius,
                                                      eye_landmarks[-1, 1]]]])
                    eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)),
                                                       'constant', constant_values=1.0))
                    eye_landmarks = (eye_landmarks *
                                     eye['inv_landmarks_transform_mat'].T)[:, :2]
                    eye_landmarks = np.asarray(eye_landmarks)
                    iris_landmarks = eye_landmarks[8:16, :]
                    iris_centre = eye_landmarks[16, :]
                    PM = ((left_eye_side_points[0][0] + left_eye_side_points[1][0])/2,(left_eye_side_points[0][1] + left_eye_side_points[1][1])/2)
                    PM -= iris_centre
                    return PM

        def _get_landmarks_start(output):
            aux = None
            for frame_index in np.unique(output['frame_index']):
                if frame_index not in data_source._frames:
                    continue
                frame = data_source._frames[frame_index]
                if 'inference' in frame['time']:
                    frame['time']['inference'] += output['inference_time']
                else:
                    frame['time']['inference'] = output['inference_time']
                aux = _get_landmarks(output)
            return aux


        def get_equation(x, y):
            return np.array([x * x, y * y, x * y, x, y, 1])

        targets = np.array([[20, 20],
                            [960, 20],
                            [1900, 20],
                            [1900, 540],
                            [960, 540],
                            [20, 540],
                            [20, 1060],
                            [960, 1060],
                            [1900, 1060]])
        # eyes = np.array([[3.29094594, 6.01488458],
        #                  [-4.23217415, 5.97621546],
        #                  [-12.64338757, 5.4388204],
        #                  [-11.06943859, 3.83121719],
        #                  [-4.90428572, 4.33292053],
        #                  [1.64777518, 4.79836256],
        #                  [0.19614697, 3.1447778],
        #                  [-11.25516758, 4.29846204],
        #                  [-13.01463917, 5.56702358]])
        # eyes = [[4.95114541,  5.74168069],
        #        [-2.89109896,  5.33137504],
        # [-9.32971376,
        # 5.88511298],
        # [-9.015723,    4.40149294],
        # [-1.93585605,
        # 3.02228478],
        # [4.68336759,  1.47680902],
        # [1.74365596, - 1.70989016],
        # [-1.33539009, - 0.9967618],
        # [-9.18661864,
        # 0.79914715]]

        eyes = [[  4.56583354,   5.4859916 ],
 [ -2.00709278,   6.27438849],
 [-10.8129257,    5.1824996 ],
 [ -9.65471203 ,  3.63369913],
 [ -2.59882551,   3.29786636],
 [  3.50023652,   2.81191832],
 [  2.1471348 ,   0.61357681],
 [ -3.57662656 ,  1.65889425],
 [ -7.76905314,   2.61117997]]

        poly = []
        for eye in eyes:
            poly.append(get_equation(eye[0], eye[1]))
        poly = np.asarray(poly)

        # Do inference forever
        infer = model.inference_generator()
        coeffsX = np.linalg.pinv(poly).dot(targets[:, 0])
        coeffsY = np.linalg.pinv(poly).dot(targets[:, 1])
        matrix = np.vstack((coeffsX, coeffsY))
        result = []
        while True:
            output = next(infer)
            pm = _get_landmarks_start(output)
            if pm is None:
                continue
            gaze = matrix.dot(get_equation(pm[0], pm[1]))
            print(gaze)
            win32api.SetCursorPos((int(gaze[0]), int(gaze[1])))


            # result.append(gaze)
        # x = np.asarray([0, 1920])
        # y = np.asarray([0,1080])
        # plt.plot(x, y)
        # for x in result:
        #     plt.plot(x[0], x[1], 'o')
        # plt.show()







