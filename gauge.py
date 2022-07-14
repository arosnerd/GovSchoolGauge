import cv2
import numpy as np
import pandas as pd
from imutils import contours
import matplotlib.pyplot as plt
import time
start_time = time.time()

image = cv2.imread("screenshot3.png")
cv2.imshow('Input', image)
cv2.waitKey(0)
#fig, ax = plt.subplots(figsize=(6, 6))

#Reduces noise, converts image to HSV, and masks black.
image = cv2.GaussianBlur(image, (5, 5), 0)
img_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
#Sets the upper and lower bounds for the mask
lower_black, upper_black = np.array([0, 0, 0]), np.array([180, 255, 65])
mask = cv2.inRange(img_hsv, lower_black, upper_black)
cv2.imshow('Mask',mask)
cv2.waitKey(0)

#Blurs mask for better circle detection
#TODO: less manual blur parameter
blurred = cv2.blur(~mask, (20, 20))
cv2.imshow('Blurred Mask',blurred)
cv2.waitKey(0)

#Converts Gaussian blurred image to grayscale and applies a threshold to find contours and draw them on the image
grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
_, thresh = cv2.threshold(grey, 140, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#iterates through and draws contours
cont_out = image.copy()
for contour in contours:
    approx = cv2.approxPolyDP(
        contour, 0.01 * cv2.arcLength(contour, True), True)
    cv2.drawContours(cont_out, [contour], 0, (0, 0, 255), 5)

blurred2 = cv2.blur(thresh, (20, 20))
cv2.imshow('Blur Threshold',blurred2)
cv2.waitKey(0)   
cv2.imshow('Threshold',thresh)
cv2.waitKey(0)   
cv2.imshow('Contours', cont_out)
cv2.waitKey(0)

#Uses Hough Circles to find the circle at the center of 
output = image.copy()
rows = blurred.shape[0]
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=30, minRadius=0, maxRadius=0)
if circles is not None:
    print('circles')
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
cv2.imshow("detected circles", output)
cv2.waitKey(0)

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
imcopy = output.copy()
blank = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.circle(blank, (circ_col, circ_row), circ_rad, 255, thickness=1) # Draw function wants center point in (col, row) order like coordinates
cv2.circle(imcopy, (circ_col, circ_row), circ_rad, 255, thickness=1)
ind_row, ind_col = np.nonzero(blank)
b = image[:, :, 0][ind_row, ind_col]
g = image[:, :, 1][ind_row, ind_col]
r = image[:, :, 2][ind_row, ind_col]
colors = list(zip(b, g, r))
cv2.imshow('blank', imcopy)
cv2.waitKey(0)

#"reverse" the row indices to get a right-handed frame of reference with origin in bottom left of image
ind_row_rev = [image.shape[0] - row for row in ind_row]
circ_row_rev = image.shape[0] - circ_row

# Convert from indexes in (row, col) order to coordinates in (col, row) order
circ_x, circ_y = circ_col, circ_row_rev
original_coord = list(zip(ind_col, ind_row_rev))

# Translate coords from gauge centers in order to compute angle between points on the perimeter
translated = []
for (x, y) in original_coord:
    translated.append((x - circ_x, y - circ_y))

# Draw lines between gauge center and perimeter pixels and compute mean and std dev of pixels along lines 
stds = []
means = []
gray_values = []

for (pt_col, pt_row_rev) in df["orig"].values:
    pt_row = -(pt_row_rev - image.shape[0])
    blank = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.line(blank, (x, y), (pt_col, pt_row), 255, thickness=2)  # Draw function wants center point in (col, row) order like coordinates
    ind_row, ind_col = np.nonzero(blank)
    b = image[:, :, 0][ind_row, ind_col]
    g = image[:, :, 1][ind_row, ind_col]
    r = image[:, :, 2][ind_row, ind_col]
    grays = (b.astype(int) + g.astype(int) + r.astype(int))/3  # Compute grayscale with naive equation
    stds.append(np.std(grays))
    means.append(np.mean(grays))
    gray_values.append(grays)

#print("Process finished --- %s seconds ---" % (time.time() - start_time))

#below is meant for full image

#creates saturated image
'''
'''
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")
(h, s, v) = cv2.split(imghsv)
s = s*1.5
s = np.clip(s,0,255)
imghsv = cv2.merge([h,s,v])
imgrgb = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)
cv2.imshow('satur',imgrgb)
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