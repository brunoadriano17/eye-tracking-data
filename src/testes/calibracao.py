
############################ IMPORTS ############################################
import math
import numpy as np
import IAMLTools
import imutils
import BlobProperties
import pygame
import ctypes
import time
import os
import matplotlib.pyplot as plt
import threading
import argparse
import queue
import coloredlogs
import cv2 as cv
import tensorflow as tf
from datasources import Video, Webcam
from models import ELG
import util.gaze

############################## DETECÇÂO DE PUPILA ######################################
class vetorEyes:
    eyes = []

    def setVet(self, x, y):
        self.eyes.append([int(x), int(y)])
        #print(self.eyes)

    def getVet(self):
        return self.eyes

    def tamanho(self):
        return self.eyes.__len__()

################################ FUNCTIONS #####################################
def onValuesChange(self, dummy=None):
    """ Handle updates when slides have changes."""
    global trackbarsValues
    trackbarsValues["threshold"] = cv2.getTrackbarPos("threshold", "Trackbars")
    trackbarsValues["minimum"]   = cv2.getTrackbarPos("minimum", "Trackbars")
    trackbarsValues["maximum"]   = cv2.getTrackbarPos("maximum", "Trackbars")

def showDetectedPupil(image, threshold, ellipses=None, centers=None, bestPupilID=None):
    """"
    Given an image and some eye feature coordinates, show the processed image.
    """
    # Copy the input image.
    eyes = vetorEyes()
    global done
    done = False

    processed = image.copy()
    if (len(processed.shape) == 2):
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    # Draw the best pupil candidate:
    if (bestPupilID is not None and bestPupilID != -1):
        pupil = ellipses[bestPupilID]
        center = centers[bestPupilID]

        cv2.ellipse(processed, pupil, (0, 255, 0), 2)

        if center[0] != -1 and center[1] != -1:
            cv2.circle(processed, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)
            #if pause == False :
            eyes.setVet(int(center[0]), int(center[1]))

            if (eyes.tamanho() >= 540):
                print(" 540 indices ok ")
                done = True
        if done == True:

            print("Acabou o laço")
            global vetEyes
            vetEyes = eyes.getVet()


    # Show the processed image.
    cv2.imshow("Detected Pupil", processed)


def detectPupil(image, threshold=101, minimum=5, maximum=50):
    """
    Given an image, return the coordinates of the pupil candidates.
    """
    # Create the output variable.
    bestPupilID = -1
    ellipses = []
    centers = []
    area = []

    kernel = np.ones((5, 5), np.uint8)

    # Grayscale image.
    grayscale = image.copy()
    if len(grayscale.shape) == 3:
        grayscale = cv2.cvtColor(grayscale, cv2.COLOR_BGR2GRAY)

    # Define the minimum and maximum size of the detected blob.
    minimum = int(round(math.pi * math.pow(minimum, 2)))
    maximum = int(round(math.pi * math.pow(maximum, 2)))

    blur = cv2.bilateralFilter(grayscale, 9, 40, 40)

    # Create a binary image.
    _, thres = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY_INV)

    cls = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find blobs in the input image.
    contours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    minRect = [None] * len(contours)
    minEllipse = [None] * len(contours)
    BestCircularity = 0

    for cnt in contours:
        prop = IAMLTools.getContourProperties(cnt, properties=["Area", "Centroid"])

        if len(cnt) > 5:
            ellipse = cv2.fitEllipse(cnt)
        else:
            ellipse = cv2.minAreaRect(cnt)

        area = prop["Area"]
        center = prop["Centroid"]

        if (area < minimum or area > maximum):
            continue

        ellipses.append(ellipse)
        centers.append(center)

        prop = IAMLTools.getContourProperties(cnt, ["Circularity"])
        circularity = prop["Circularity"]
        curva = cv2.arcLength(cnt, True)

        if (abs(1. - circularity) < abs(1. - BestCircularity)):

            BestCircularity = circularity
            bestPupilID = len(ellipses) - 1

            if (BestCircularity == circularity):
                if ((area > 3000 and area < 3900) or curva < 300):
                    pass
                    #showDetectedPupil(image, threshold, ellipses, centers, bestPupilID)

    # Return the final result.
    return ellipses, centers, bestPupilID
############################## FIM DETECÇÂO DE PUPILA ######################################

# Define the trackbars.
trackbarsValues = {}
trackbarsValues["threshold"] = 75
trackbarsValues["minimum"] = 13
trackbarsValues["maximum"] = 32
# trackbarsValues["area"]  = 5

# Create an OpenCV window and some trackbars.
#cv2.namedWindow("Trackbars", cv2.WINDOW_AUTOSIZE)
#cv2.createTrackbar("threshold", "Trackbars", 0, 255, onValuesChange)
#cv2.createTrackbar("minimum", "Trackbars", 5, 40, onValuesChange)
#cv2.createTrackbar("maximum", "Trackbars", 50, 100, onValuesChange)
# cv2.createTrackbar("area",  "Trackbars",  5, 400, onValuesChange)

#cv2.imshow("Trackbars", np.zeros((3, 500), np.uint8))

# Create a capture video object.
filename = "inputs/teste.mp4"
capture = cv2.VideoCapture(filename)
acabou = False

threadLock = threading.Lock()
threads = []
# tela cheia
os.environ['SDL_VIDEO_CENTERED'] = '1'
pygame.init()

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

#GET POSITION
def _getPupilVector():
    vet = []
    carregado = True
    x = 0
    while carregado:
    # Capture frame-by-frame.
        retval, frame = capture.read()

        # Check if there is a valid frame.
        if not retval:
            # Restart the video.
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Get the detection parameters values.
        threshold = trackbarsValues["threshold"]
        minimum = trackbarsValues["minimum"]
        maximum = trackbarsValues["maximum"]

        # Pupil detection.
        ellipses, centers, bestPupilID = detectPupil(frame, threshold, minimum, maximum)
        # Draw the best pupil candidate:
        if (bestPupilID is not None and bestPupilID == 1):
            pupil = ellipses[bestPupilID]
            center = centers[bestPupilID]
            if center[0] != -1 and center[1] != -1:
                vet.append([int(center[0]), int(center[1])])
                x += 1
        if(x == 60):
            carregado = False
    return vet

class vetorTargets:
    target = []

    def setVet(self, px, py):
        self.target.append([int(px), int(py)])
        #print(self.target)

    def getVet(self):
        return self.target

## INICIO ANIMAÇÃO COM DETECÇÃO DA PUPILA ##
global vet
global pause
i = 0
py = 0
px = 0
direita = True
voltar = False
done = False
vet = vetorTargets()
vetFinal = []
for x in range(9):
    if(i == 0):
        px = 0
        py = 0
    if(i == 1):
        px += nextx
    if(i == 2):
        px += nextx
    if (i == 3):
        py += nexty
    if (i == 4):
        px -= nextx
    if (i == 5):
        px = 0
    if (i == 6):
        py += nexty
    if (i == 7):
        px += nextx
    if (i == 8):
        px += nextx
    screen.fill(BLACK)
    pygame.draw.rect(screen, (255, 255, 0), [px, py, 40, 40])
    pygame.display.flip()
    time.sleep(2)
    #CAPTURA 60 POSIÇÕES NO PONTO ATUAL DA ANIMAÇÃO
    vetFinal.append(_getPupilVector())
    i+= 1;
pygame.quit()
global vetTarget
vetTarget = vet.getVet()
print(vetFinal)

