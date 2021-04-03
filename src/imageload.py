import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

import os

# plt.style.use('ggplot')
# x = ['24\'\'', '21\'\'', '19\'\'', '14\'\'']
# energy = []
# arr = []
# for z in range(1, 41):
#     aux = open('./calib_out/data/'+str(z)+'_data.txt')
#     media = aux.read()
#     media = media.split('Media DP:',1)[1]
#     arr.append(float(media))
#     if len(arr) == 10:
#         energy.append(np.mean(np.asarray(arr)))
#         arr = []
# print(energy)
#
# x_pos = [i for i, _ in enumerate(x)]
# plt.bar(x_pos, energy, color='blue')
# plt.xlabel("Monitores")
# plt.ylabel("Erro medio em graus")
# plt.title("Média DP em 10 amostras/tamanho de tela")
# plt.xticks(x_pos, x)
#
# plt.show()

plt.style.use('ggplot')
x = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
pos = np.empty((10, 9), float)
arr = []
energy = []
j = 0
for z in range(41, 51):
    print(z)
    aux = open('./calib_out/data/'+str(z)+'_data.txt')
    media = aux.read()
    media = media.split('em graus:',1)[1]
    media = media.split('Media DP', 1)[0]
    media = media.replace('[', '')
    media = media.replace('array(', '')
    media = media.replace(')', '')
    media = media.replace(']', '')
    vet = media.split(',')
    for y in range(0, 9):
        pos[j][y] = float(vet[y])
    j = j+1

energy.append(np.mean(pos[:, 0]))
energy.append(np.mean(pos[:, 1]))
energy.append(np.mean(pos[:, 2]))
energy.append(np.mean(pos[:, 3]))
energy.append(np.mean(pos[:, 4]))
energy.append(np.mean(pos[:, 5]))
energy.append(np.mean(pos[:, 6]))
energy.append(np.mean(pos[:, 7]))
energy.append(np.mean(pos[:, 8]))


fig = plt.figure()

ax = Axes3D(fig) #<-- Note the difference from your original code...
cset = ax.bar3d([0, 0, 0, 540, 540, 540, 1080, 1080, 1080], [0, 960, 1920, 1920, 960, 0, 0, 960, 1920],[0, 0, 0,0, 0, 0,0, 0, 0], [100, 100, 100, 100, 100, 100,100, 100, 100], [100, 100, 100, 100, 100, 100,100, 100, 100], energy, alpha=0.2, color='blue')
plt.gca().invert_xaxis()
ax.text(0, 0, 0, str(energy[0])[0:5]+"°", color='red')
ax.text(0, 960, 0, str(energy[1])[0:5]+"°", color='red')
ax.text(0, 1920, 0, str(energy[2])[0:5]+"°", color='red')
ax.text(540, 1920, 0, str(energy[3])[0:5]+"°", color='red')
ax.text(540, 960, 0, str(energy[4])[0:5]+"°", color='red')
ax.text(540, 0, 0, str(energy[5])[0:5]+"°", color='red')
ax.text(1080, 0, 0, str(energy[6])[0:5]+"°", color='red')
ax.text(1080, 960, 0, str(energy[7])[0:5]+"°", color='red')
ax.text(1080, 1920, 0, str(energy[8])[0:5]+"°", color='red')

plt.xlabel("Y")
plt.ylabel("X")
plt.title("Monitor: 14\'' | Erro médio: "+str(np.mean(pos))[0:5]+"°")
plt.show()