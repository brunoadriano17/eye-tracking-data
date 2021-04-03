import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import ctypes

def get_equation(x, y):
    return np.array([x * x, y * y, x * y, x, y, 1])

user32 = ctypes.windll.user32

# 1366 768- tamanho da minha tela
# sizeX = user32.GetSystemMetrics(0)
# sizeY = user32.GetSystemMetrics(1)
sizeX = 1366
sizeY = 768
vetTargets = [[20, 20], [(sizeX/2), 20], [sizeX, 20], [sizeX, (sizeY/2)], [(sizeX/2), (sizeY/2)], [20, (sizeY/2)], [20, sizeY], [(sizeX/2), sizeY], [sizeX, sizeY]]
targets = np.asarray(vetTargets)
plt.style.use('ggplot')

tot20 = 0
tot40 = 0
tot60 = 0
tot80 = 0
tot100 = 0
tot120 = 0
tot140 = 0
tot160 = 0

for z in range(31, 41):
    aux = open('./calib_out/data/'+str(z)+'_data.txt')
    texto = aux.read()

    pontos = texto.split('540 pontos:[', 1)[1]
    pontos = pontos.split('Média/60', 1)[0]
    pontos = pontos.replace('[', '')
    pontos = pontos.replace('array(', '')
    pontos = pontos.replace(')', '')
    pontos = pontos.replace(']', '')
    vet = pontos.split(',')
    coords = []
    vetMedia = []
    x = None
    y = None
    for i in vet:
        if x == None:
            x = float(i)
        else:
            y = float(i)
            coords.append([x, y])
            x = None
            y = None

    media = texto.split('Média/60 pontos:')[1]
    media = media.split('Matriz de')[0]
    media = media.replace('[', '')
    media = media.replace(']', '')
    media = media.split(',')
    x = None
    y = None
    for i in media:
        if x == None:
            x = float(i)
        else:
            y = float(i)
            vetMedia.append([x, y])
            x = None
            y = None

    poly = []
    for eye in vetMedia:
        poly.append(get_equation(eye[0], eye[1]))
    poly = np.asarray(poly)

    coeffsX = np.linalg.pinv(poly).dot(targets[:, 0])
    coeffsY = np.linalg.pinv(poly).dot(targets[:, 1])
    matrix = np.vstack((coeffsX, coeffsY))
    j = 0
    count = 0
    count2 = 0
    vetAux = []

    # plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    # plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    #
    # for i in targets:
    #     plt.plot(i[0], i[1], 'bs', markersize=20)

    for i in coords:
        gaze = matrix.dot(get_equation(i[0], i[1]))
        count = count + 1
        if (count%61) == 0:
            j = j+1
        if abs((targets[j][0] - gaze[0])) <= 20 and abs(targets[j][1] - gaze[1]) <= 20:
            # plt.plot(gaze[0], gaze[1], 'ro')
            tot20 = tot20+1
        if abs((targets[j][0] - gaze[0])) <= 40 and abs(abs(targets[j][1] - gaze[1])) <= 40:
            # plt.plot(gaze[0], gaze[1], 'ro')
            tot40 = tot40+1
        if abs((targets[j][0] - gaze[0])) <= 60 and abs(targets[j][1] - gaze[1]) <= 60:
            # plt.plot(gaze[0], gaze[1], 'ro')
            tot60 = tot60+1
        if abs((targets[j][0] - gaze[0])) <= 80 and abs(targets[j][1] - gaze[1]) <=80:
            # plt.plot(gaze[0], gaze[1], 'ro')
            tot80 = tot80+1
        if abs((targets[j][0] - gaze[0])) <= 100 and abs(targets[j][1] - gaze[1]) <= 100:
            # plt.plot(gaze[0], gaze[1], 'ro')
            tot100 = tot100+1
        if abs((targets[j][0] - gaze[0])) <= 120 and abs(targets[j][1] - gaze[1]) <= 120:
            # plt.plot(gaze[0], gaze[1], 'ro')
            tot120 = tot120+1
        if abs((targets[j][0] - gaze[0])) <= 140 and abs(targets[j][1] - gaze[1]) <= 140:
            # plt.plot(gaze[0], gaze[1], 'ro')
            tot140 = tot140+1
        if abs((targets[j][0] - gaze[0])) <= 160 and abs(targets[j][1] - gaze[1]) <= 160:
            # plt.plot(gaze[0], gaze[1], 'ro')
            tot160 = tot160+1
    plt.gca().invert_yaxis()



x = ['20px', '40px', '60px', '800px', '100px', '120px', '140px', '160px']
x_pos = [i for i, _ in enumerate(x)]
energy = [tot20*100/5400, tot40*100/5400, tot60*100/5400, tot80*100/5400, tot100*100/5400, tot120*100/5400, tot140*100/5400, tot160*100/5400]
plt.bar(x_pos, energy, color='blue')
plt.ylabel("% Acerto")
plt.xticks(x_pos, x)
print(tot20*100/5400)
print(tot160*100/5400)
plt.show()