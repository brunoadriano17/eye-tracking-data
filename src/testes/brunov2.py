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
import pygame
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

    parser.add_argument('--fps', type=int, default=60, help='Desired sampling rate of webcam')
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
    pygame.init()
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

        def _get_iris(output):


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
            result = []
            for frame_index in np.unique(output['frame_index']):
                if frame_index not in data_source._frames:
                    continue
                frame = data_source._frames[frame_index]
                if 'inference' in frame['time']:
                    frame['time']['inference'] += output['inference_time']
                else:
                    frame['time']['inference'] = output['inference_time']
                aux = _get_landmarks(output)
                if(len(aux) > 0):
                    result.append(aux)
            return result

        # iniciar generator
        # infer = model.inference_generator()
        frames = data_source.frame_generator()
        # ## calibracao ##
        # # tela cheia
        os.environ['SDL_VIDEO_CENTERED'] = '1'

        # Define algumas cores colors
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)

        # Seta a largura e a altura da tela utilizada [width, height](só funciona para windows)
        user32 = ctypes.windll.user32

        # 1366 768- tamanho da minha tela
        sizeX = user32.GetSystemMetrics(0)
        sizeY = user32.GetSystemMetrics(1)

        size = sizeX, sizeY
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        #pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

        nextx = int(sizeX / 2) - 20
        nexty = int(sizeY / 2) - 20

        # nome da janela
        pygame.display.set_caption("Calibração")

        currentX = 0
        currentI = 0
        vetFinal = []
        vetFrames = []
        run = True
        while run:
            pygame.event.pump()
            done = False
            if(currentI == 0):
                px = 0
                py = 0
            if(currentI == 1):
                px += nextx
            if(currentI == 2):
                px += nextx
            if (currentI == 3):
                py += nexty
            if (currentI == 4):
                px -= nextx
            if (currentI == 5):
                px = 0
            if (currentI == 6):
                py += nexty
            if (currentI == 7):
                px += nextx
            if (currentI == 8):
                px += nextx

            screen.fill(BLACK)
            pygame.draw.rect(screen, (255, 255, 0), [px, py, 40, 40])
            #pygame.draw.rect(screen, (255, 255, 0), [940, 520, 40, 40])
            pygame.display.flip()

            ## TEST ONE POSITION ##
            # for x in range(60):
            #     print('x: ', x)
            #     output = next(infer)
            #     aux = _get_landmarks_start(output)
            #     if (len(aux) > 0):
            #         vetFinal.append(aux)
            #     else:
            #         x -= 1
            #
            # if(x == 59):
            #     run = False
            ## END TEST ONE POSITION ##

            #CAPTURA 60 POSIÇÕES NO PONTO ATUAL DA ANIMAÇÃO
            print('processing...')
            for x in range(60):
                # output = next(infer)
                output = next(frames)
                print('output: ')
                vetFrames.append(output)
                aux = _get_landmarks_start(output)
                if(len(aux) > 0):
                    vetFinal.append(aux)
                else:
                    x -= 1
            if(len(vetFinal) == 60*(currentI+1)):
                currentI += 1

            if(currentI == 9):
                run = False

        pygame.quit()
        print('vetor preenchido...')
        print(vetFinal)

        ##FIM-----##
        # print(vetFrames)
        # print("finish...")
        # print('process frames...')
        # frame_infer = model.inference_fetches_generated(vetFrames)
        # run = True
        # for f in vetFrames:
        #     vetFinal.append(_get_landmarks_start(f))
        #     print('vetFinal running...')
        #     if len(vetFinal) == 540:
        #         run = False
        # print('vetFinal done...')
        # print(vetFinal)
        # vetNp = np.asarray(vetFinal)
        # print(vetNp)

        # MOVER MOUSER #
        # px = 0.0;
        # py = 0.0;
        # for x in range(60):
        #     px += vetFinal[x][0][0]
        #     py += vetFinal[x][0][1]
        #
        # pmx = px / 60
        # pmy = py / 60
        #
        # while True:
        #     next_fr = next(infer)
        #     current_pos = [_get_landmarks_start(next_fr)]
        #     newMouseX = current_pos[0][0][0]*980/pmx
        #     newMouseY = current_pos[0][0][1]*540/pmy
        #     print('MX: ', pmx)
        #     print('MY: ', pmy)
        #     print('currentIrisX: ', current_pos[0][0][0])
        #     print('currentIrisY: ', current_pos[0][0][1])
        #     print('X: ', newMouseX)
        #     print('Y: ', newMouseY)
        #     win32api.SetCursorPos((int(newMouseX), int(newMouseY)))


        # FIM MOVER MOUSE #

        # p1x = 0
        # p1y = 0
        # p2x = 0
        # p2y = 0
        # p3x = 0
        # p3y = 0
        # p4x = 0
        # p4y = 0
        # p5x = 0
        # p5y = 0
        # p6x = 0
        # p6y = 0
        # p7x = 0
        # p7y = 0
        # p8x = 0
        # p8y = 0
        # p9x = 0
        # p9y = 0
        # for x in range(540):
        #     if 0 <= x < 60:
        #         p1x += int(vetFinal[x][0][0])
        #         p1y += int(vetFinal[x][0][1])
        #     if 60 <= x < 120:
        #         p2x += int(vetFinal[x][0][0])
        #         p2y += int(vetFinal[x][0][1])
        #     if 120 <= x < 180:
        #         p3x += int(vetFinal[x][0][0])
        #         p3y += int(vetFinal[x][0][1])
        #     if 180 <= x < 240:
        #         p4x += int(vetFinal[x][0][0])
        #         p4y += int(vetFinal[x][0][1])
        #     if 240 <= x < 300:
        #         p5x += int(vetFinal[x][0][0])
        #         p5y += int(vetFinal[x][0][1])
        #     if 300 <= x < 360:
        #         p6x += int(vetFinal[x][0][0])
        #         p6y += int(vetFinal[x][0][1])
        #     if 360 <= x < 420:
        #         p7x += int(vetFinal[x][0][0])
        #         p7y += int(vetFinal[x][0][1])
        #     if 420 <= x < 480:
        #         p8x += int(vetFinal[x][0][0])
        #         p8y += int(vetFinal[x][0][1])
        #     if 480 <= x < 540:
        #         p9x += int(vetFinal[x][0][0])
        #         p9y += int(vetFinal[x][0][1])
        #
        # print('media p1: ', (p1x/9, p1y/9))
        # print('media p2: ', (p2x / 9, p2y / 9))
        # print('media p3: ', (p3x / 9, p3y / 9))
        # print('media p4: ', (p4x / 9, p4y / 9))
        # print('media p5: ', (p5x / 9, p5y / 9))
        # print('media p6: ', (p6x / 9, p6y / 9))
        # print('media p7: ', (p7x / 9, p7y / 9))
        # print('media p8: ', (p8x / 9, p8y / 9))
        # print('media p9: ', (p9x / 9, p9y / 9))



