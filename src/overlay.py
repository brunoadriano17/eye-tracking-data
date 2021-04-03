import cv2 as cv
import numpy as np

img = cv.imread('D:/2.png');
img_transp = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
height, width, channels = img.shape
firstX, lastX, firstY, lastY = -1, -1, -1, -1
for i in range(0, height):
    for j in range(0, width):
        (b, g, r, a) = img_transp[i][j]
        if b == 0 and g >= 240 and r == 0:
            img_transp[i][j][0] = 255
            img_transp[i][j][1] = 255
            img_transp[i][j][2] = 255
            img_transp[i][j][3] = 0
            if firstX == -1:
                firstX = i
                firstY = j
                lastX = i
                lastY = j
            else:
                if i < firstX:
                    firstX = i
                if i > lastX:
                    lastX = i
                if j < firstY:
                    firstY = j
                if j > lastY:
                    lastY = j

w, h = lastY - firstY, lastX-firstX


cv.imwrite('D:/outpy.png', img_transp[firstX:lastX, firstY:lastY])

cv.waitKey(0)
cv.destroyAllWindows()