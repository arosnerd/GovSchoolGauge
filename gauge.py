import cv2
import numpy as np
import pandas as pd
from imutils import contours
import matplotlib.pyplot as plt
import time
start_time = time.time()

#read in initial image
image = cv2.imread("screenshot.png")

#Reduces noise, converts image to HSV
image = cv2.GaussianBlur(image, (5, 5), 0)
img_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

#Sets the upper and lower bounds for the mask, and masks black
lower_black, upper_black = np.array([0, 0, 0]), np.array([180, 255, 65])
mask = cv2.inRange(img_hsv, lower_black, upper_black)

#Blurs mask for better circle detection
#TODO: less manual blur parameter
blurred = cv2.blur(~mask, (20, 20))

#Converts Gaussian blurred image to grayscale and applies a threshold to find contours and draw them on the image
grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
_, thresh = cv2.threshold(grey, 140, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#iterates through list of contours and draws contours
cont_out = image.copy()
for contour in contours:
    approx = cv2.approxPolyDP(
        contour, 0.01 * cv2.arcLength(contour, True), True)
    cv2.drawContours(cont_out, [contour], 0, (0, 0, 255), 5)

#Uses Hough Circles to find the circle at the center of needle
output = image.copy()
rows = blurred.shape[0]
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=30, minRadius=0, maxRadius=0)
if circles is not None:
    circles = np.uint16(np.around(circles))
    i = 0
    for i in circles[0, :]:
        center = (i[0], i[1])
        circ_col = center[0]
        circ_row = center[1]
        # circle center
        cv2.circle(output, center, 1, (255, 255, 0), 3)
        # circle outline
        radius = i[2]
        circ_rad = radius*7
        cv2.circle(output, center, radius, (0, 0, 255), 3)
# cv2.imshow("detected circles", output)
# cv2.waitKey(0)

test = output.copy()
blank = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.circle(blank, (circ_col, circ_row), circ_rad, 255, thickness=1)
ind_row, ind_col = np.nonzero(blank)
ind_row = np.array(ind_row)
ind_col = np.array(ind_col)
for col,row in zip(ind_col, ind_row):
        cv2.circle(blank, (col,row), 1, 255, thickness=1)
        cv2.circle(test, (col,row), 1, 255, thickness=1)
# cv2.imshow('blank', blank)
# cv2.waitKey(0)
# cv2.imshow('blank', test)
# cv2.waitKey(0)

imcopy = image.copy()
stds = []
means = []
gray_values = []
i = 0
for col,row in zip(ind_col, ind_row):
    blank = np.zeros(image.shape[:2], dtype=np.uint8)
    if i%5 == 0:
        cv2.line(imcopy, (circ_col, circ_row), (col,row), 255, thickness=1)
    i = i+1
    cv2.line(blank, (circ_col, circ_row), (col, row), 255, thickness=2)  # Draw function wants center point in (col, row) order like coordinates
    ind_row, ind_col = np.nonzero(blank)
    b = image[:, :, 0][ind_row, ind_col]
    g = image[:, :, 1][ind_row, ind_col]
    r = image[:, :, 2][ind_row, ind_col]
    grays = (b.astype(int) + g.astype(int) + r.astype(int))/3  # Compute grayscale with naive equation
    stds.append(np.std(grays))
    means.append(np.mean(grays))
    gray_values.append(grays)
print("Process finished --- %s seconds ---" % (time.time() - start_time))        
# cv2.imshow('lines',imcopy)
# cv2.waitKey(0)

#show all plots at once
cv2.imshow('Input', image)
cv2.waitKey(0)

cv2.imshow('Mask',mask)
cv2.waitKey(0)

cv2.imshow('Blurred Mask',blurred)
cv2.waitKey(0)
  
cv2.imshow('Threshold',thresh)
cv2.waitKey(0)   
cv2.imshow('Contours', cont_out)
cv2.waitKey(0)

cv2.imshow("detected circles", output)
cv2.waitKey(0)

cv2.imshow('blank', blank)
cv2.waitKey(0)
cv2.imshow('blank', test)
cv2.waitKey(0)

cv2.imshow('lines',imcopy)
cv2.waitKey(0)
'''
minLineLength = 60
maxLineGap = 0
lines = cv2.HoughLinesP(image=blurred, rho=50, theta=np.pi / 180, threshold=100,minLineLength=minLineLength, maxLineGap=0)  # rho is set to 3 to detect more lines, easier to get more then filter them out later
#for testing purposes, show all found lines
for i in range(0, len(lines)):
    for x1, y1, x2, y2 in lines[i]:
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow("lines",output)
cv2.waitKey(0)
'''

#filtering red
''''
mask = cv2.inRange(imghsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
mask2 = cv2.inRange(imghsv, np.array([170, 70, 50]), np.array([180, 255, 255]))

finalMask = mask | mask2
result = cv2.bitwise_and(image, image, mask = finalMask)
noRed = image - result
noRedRgb = cv2.cvtColor(noRed, cv2.COLOR_HSV2RGB)

cv2.imshow('mask', result)
cv2.waitKey(0)
cv2.imshow('nored', noRed)
cv2.waitKey(0)
'''