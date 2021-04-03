import argparse
import numpy as np
import coloredlogs
import tensorflow as tf
from datasources import Video, Webcam
from models import ELG
import cv2 as cv
import pygame
import ctypes
import time
import os
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

captura = cv.VideoCapture(0)
# Inicia sessão
pygame.init()
## calibracao ##
# tela cheia
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
# Do inference forever
# frames = data_source.frame_generator()

currentX = 0
currentI = 0
vetFinal = []
vetPontoMedio = []
run = True
while run:
    pygame.event.pump()
    global vet
    global pause
    i = 0
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
    pygame.display.flip()
    time.sleep(1)
    ready = False
    #CAPTURA 60 POSIÇÕES NO PONTO ATUAL DA ANIMAÇÃO
    for x in range(60):
        ret, frame = captura.read()
        if ret:
            vetFinal.append(frame)
        else:
            x -= 1
    print('next...')
    currentI += 1
    # if(len(vetFinal) == 60*(currentI+1)):
    #     currentI += 1

    if(currentI == 9):
        run = False

pygame.quit()
print('vetSize ', len(vetFinal))
# Finaliza captura da camera #
captura.release()

## vetFinal contem 540 frames da calibração ##
## processar os frames agora para obter os pontos ##
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

    # Carrega vetor para classe
    data_source = Video(vetFinal,
                        tensorflow_session=session, batch_size=batch_size,
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
                # retorna landmarks das laterais dos olhos esquerdo e direito até [f][:-1]
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
                PM = ((left_eye_side_points[0][0] + left_eye_side_points[1][0]) / 2,
                      (left_eye_side_points[0][1] + left_eye_side_points[1][1]) / 2)
                PM -= iris_centre
                cv.imshow('eye', frame)
                return iris_centre


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
            if (len(aux) > 0):
                result.append(aux)
        return result

    infer = model.inference_generator()
    print('starting calibracao...')
    for x in range(1):
        print('x: ', x)
        output = next(infer)
        aux = _get_landmarks_start(output)
        vetPontoMedio.append(aux)

    print('frames processados: ', len(vetPontoMedio))

    vetMediaFinal = []
    j = 0
    pmX = 0
    pmY = 0
    for i in range(540):
        pmX += vetPontoMedio[i][0][0]
        pmY += vetPontoMedio[i][0][1]
        j += 1
        if j == 60:
            j = 0
            auxPmX = pmX / 60;
            auxPmY = pmY / 60;
            vetMediaFinal.append([auxPmX, auxPmY])
            pmX = 0
            pmY = 0

    print('vet com media por ponto processada...')
    print(vetMediaFinal)

    print('finished...')


