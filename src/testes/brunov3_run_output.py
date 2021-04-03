import argparse
import numpy as np
import coloredlogs
import tensorflow as tf
from datasources import Video, Webcam
from models import ELG
import cv2 as cv
import pandas as pd
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
import random
import win32api, win32con
import threading


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

vetFinal = []
vetMediaFinal = []

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
eyes = []
tf.logging.set_verbosity(tf.logging.INFO)
# with tf.Session(config=session_config) as session:
session = tf.Session(config=session_config)
with session:
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

    infer = model.inference_generator()
    print('starting calibracao...')
    for x in range(540):
        print('x: ', x)
        output = next(infer)
        print('x processado')
        aux = _get_landmarks_start(output)
        vetPontoMedio.append(aux)
    print(vetPontoMedio)
    print('frames processados: ', len(vetPontoMedio))

    j = 0
    pmX = 0
    pmY = 0
    for i in range(540):
        pmX += vetPontoMedio[i][0]
        pmY += vetPontoMedio[i][1]
        j += 1
        if j == 60:
            j = 0
            auxPmX = pmX / 60;
            auxPmY = pmY / 60;
            vetMediaFinal.append([auxPmX, auxPmY])
            pmX = 0
            pmY = 0

    print('vet com media por ponto processada...')

    targets = np.asarray(vetTargets)
    eyes = np.asarray(vetMediaFinal)
    print('eyes: ', eyes)
    print('targets: ', targets)

    print('calibracao finished...')

    x = np.asarray(
        [vetMediaFinal[0][0], vetMediaFinal[1][0], vetMediaFinal[2][0], vetMediaFinal[3][0], vetMediaFinal[4][0], vetMediaFinal[5][0], vetMediaFinal[6][0],
         vetMediaFinal[7][0], vetMediaFinal[8][0]])
    y = np.asarray(
        [vetMediaFinal[0][1], vetMediaFinal[1][1], vetMediaFinal[2][1], vetMediaFinal[3][1], vetMediaFinal[4][1], vetMediaFinal[5][1], vetMediaFinal[6][1],
         vetMediaFinal[7][1], vetMediaFinal[8][1]])
    # Save output plot and data
    plt.plot(x, y)
    plt.savefig(mediaFileName + '-ponto_medio.png')
    for x in vetPontoMedio:
        plt.plot(x[0], x[1], 'o')
    plt.savefig(fileName + '-ponto_medio_com_540_pontos.png')

session.close()
tf.reset_default_graph()


def get_equation(x, y):
    return np.array([x * x, y * y, x * y, x, y, 1])


poly = []
for eye in eyes:
    poly.append(get_equation(eye[0], eye[1]))
poly = np.asarray(poly)

vetTargets = np.asarray(vetTargets)
coeffsX = np.linalg.pinv(poly).dot(vetTargets[:, 0])
coeffsY = np.linalg.pinv(poly).dot(vetTargets[:, 1])
matrix = np.vstack((coeffsX, coeffsY))

##### CALCULO DESVIO PADRÃO ####
print("Iniciando calculo do desvio padrão...")
i = 1
# target index
j = 0
# vetor de 9 posições com o desvio
vetDp = []
gazeVec = []
# vetPontoMedio com 540 pontos processados
for x in vetMediaFinal:
    # calcula distancia euclidiana
    gaze = matrix.dot(get_equation(x[0], x[1]))
    # calcula a distancia x e y do ponto ao target
    dEuc = np.linalg.norm(gaze - targets[j])
    # adiciona ao vetor
    # converte array para nparray
    aux = np.asarray(dEuc)
    vetDp.append(aux)
    j = j + 1
print('Médias: ', vetDp)

# User Distance in mm
userDistance = 450
# Largura monitor in mm
monitorWidth = 410

vetGraus = []
## Conversão ##
for x in vetDp:
    print('x', x)
    result = (360 / math.pi) * math.atan((x * (monitorWidth / sizeX)) / (2 * userDistance))
    vetGraus.append(result)

print('Conversão: ', vetGraus)
medDp = np.mean(vetGraus)
print('Media DP: ', medDp)

#Save processed data to text file
outputString = '540 pontos:'+str(vetPontoMedio)+'\n\n'+'Média/60 pontos:'+str(vetMediaFinal)+'\n\n'+'Matriz de calibração:'+str(matrix)+'\n\n'+'Desvio padrão/60pontos em graus:'+str(vetGraus)+'\n\n'+'Media DP:'+str(medDp)
text_file = open(outTextFileName+'_data.txt', 'w')
n = text_file.write(outputString)
text_file.close()

## Inicia captura da câmera e movimentação do cursor ##
#
# sess = tf.Session(config=session_config)
# with sess:
#     # Parametros
#     batch_size = 2
#
#     # Carrega webcam para classe
#     data_source = Webcam(tensorflow_session=sess, batch_size=batch_size,
#                          camera_id=args.camera_id, fps=args.fps,
#                          data_format='NCHW' if gpu_available else 'NHWC',
#                          eye_image_shape=(36, 60))
#     # Modelo
#     model = ELG(
#         sess, train_data={'videostream': data_source},
#         first_layer_stride=1,
#         num_modules=2,
#         num_feature_maps=32,
#         learning_schedule=[
#             {
#                 'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']},
#             },
#         ],
#     )
#
#     ## Inicia estimativa e movimentação do cursor ##
#
#     # Do inference forever
#     infer = model.inference_generator()
#     result = []
#     last_three = []
#     while True:
#         output = next(infer)
#         pm = _get_landmarks_start(output)
#         if pm is None:
#             continue
#         gaze = matrix.dot(get_equation(pm[0], pm[1]))
#         last_three.append(gaze)
#         if len(last_three) > 10:
#             last_three.pop(0)
#         if len(last_three) == 10:
#             pmedx = 0
#             pmedy = 0
#             for x in range(10):
#                 pmedx = pmedx + last_three[x][0]
#                 pmedy = pmedy + last_three[x][1]
#             # pmedx = (last_three[0][0] + last_three[1][0] + last_three[2][0])/3
#             # pmedy = (last_three[0][1] + last_three[1][1] + last_three[2][1]) / 3
#             print('x: ', (int(pmedx/10)), ' y: ', (int(pmedy/10)))
#             win32api.SetCursorPos((int(pmedx/10), int(pmedy/10)))

outputVet = []
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
run = True
k = 0
targets2 = []
while run:
    pygame.event.pump()
    global vet
    global pause
    i = 0
    done = False
    screen.fill(BLACK)
    px = random.randint(0, sizeX)
    py = random.randint(0, sizeY)
    targets2.append([px, py])
    pygame.draw.rect(screen, (255, 255, 0), [px, py, 40, 40])
    pygame.display.flip()
    time.sleep(1)
    ready = False
    #CAPTURA 60 POSIÇÕES NO PONTO ATUAL DA ANIMAÇÃO
    for x in range(60):
        ret, frame = captura.read()
        outputVet.append(frame)

    if len(outputVet) == 360:
        run = False

pygame.quit()
print('vetSize ', len(outputVet))
# Finaliza captura da camera #
captura.release()

ses = tf.Session(config=session_config)
with ses:
    # Parametros
    batch_size = 2

    # Carrega vetor para classe
    data_source = Video(outputVet,
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

    infer = model.inference_generator()
    print('starting calibracao...')
    gazeOut = []
    gazeFinal = []
    for x in range(360):
        print('x processado')
        pm = _get_landmarks_start(output)
        if pm is None:
            continue
        gaze = matrix.dot(get_equation(pm[0], pm[1]))
        gazeOut.append(gaze)
        if (len(gazeOut) % 60) == 0:
            gazeFinal.append(np.mean(np.asarray(gazeOut)))
            gazeOut = []
    print('targets: ', targets2)
    print('gaze: ', gazeFinal)
