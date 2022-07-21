import cv2
import numpy as np
import pandas as pd
from imutils import contours
import matplotlib.pyplot as plt
import math

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

lower_black, upper_black = np.array([0, 0, 0]), np.array([180, 255, 180])
mask2 = cv2.inRange(img_hsv, lower_black, upper_black)
# cv2.imshow('mask2',mask2)
# cv2.waitKey(0)

#Blurs mask for better circle detection
#TODO: less manual blur parameter
blurred = cv2.blur(~mask, (20, 20))

grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
_, thresh = cv2.threshold(grey, 140, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#iterates through and draws contours
cont_out = image.copy()
for contour in contours:
    approx = cv2.approxPolyDP(
        contour, 0.01 * cv2.arcLength(contour, True), True)
    cv2.drawContours(cont_out, [contour], 0, (0, 0, 255), 5)

#Uses Hough Circles to find the circle at the center of needle using the black mask, and draws it
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
        cv2.circle(output, center, radius, (0, 0, 255), 3)
        circ_rad = radius*8

blank = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.circle(blank, (circ_col, circ_row), circ_rad, 255, thickness=1)
ind_row, ind_col = np.nonzero(blank)
ind_row = np.array(ind_row)
ind_col = np.array(ind_col)
for col,row in zip(ind_col, ind_row):
        cv2.circle(blank, (col,row), 1, 255, thickness=1)

df = pd.DataFrame({"indices":list(zip(ind_col, ind_row))})
first_col = ind_col[0]
first_row = ind_row[0]
top_dist = math.sqrt((circ_col-first_col)**2 + (circ_row-first_row)**2)
stds = []
means = []
gray_values = []
# stds_end = []
# means_end = []
# gray_values_end = []
angles = []
blkis = []
blkes = []
i = 0
lines = image.copy()
tester = image.copy()
for col,row in zip(ind_col, ind_row):
    blki = 0
    blke = 0
    blank = np.zeros(image.shape[:2], dtype=np.uint8)
    
    #angle calculation
    top_point_dist = math.sqrt((col-first_col)**2 + (row-first_row)**2)
    angle = math.acos((((2*top_dist**2)-top_point_dist**2)/(2*top_dist**2)))
    angle = angle*(180/(math.pi))
    if col < first_col:
        angle = (angle * -1)+360
    angles.append(angle)
    
    #visualization
    if i%5 == 0:
        cv2.line(lines, (circ_col, circ_row), (col,row), 255, thickness=1)
    i = i+1
    cv2.line(blank, (circ_col, circ_row), (col, row), 255, thickness=2)  # Draw function wants center point in (col, row) order like coordinates
    ind_row, ind_col = np.nonzero(blank)

    h = []
    s = []
    v = []
    b = []
    g = []
    r = []

    itercount = 0
    for col,row in zip(ind_col, ind_row):
        if row > circ_row:
            h.insert(0, img_hsv[row, col, 0])
            s.insert(0, img_hsv[row, col, 1])
            v.insert(0, img_hsv[row, col, 2])
            b.insert(0, image[row, col, 0])
            g.insert(0, image[row, col, 1])
            r.insert(0, image[row, col, 2])
        else:
            h.append(img_hsv[row, col, 0])
            s.append(img_hsv[row, col, 1])
            v.append(img_hsv[row, col, 2])
            b.append(image[row, col, 0])
            g.append(image[row, col, 1])
            r.append(image[row, col, 2])
        if itercount > round(len(ind_col)*0.65) and itercount < round(len(ind_col)*0.85):
            print('test')
            cv2.circle(tester, (col, row), 1, (0, 0, 255), thickness=3)
        itercount += 1

    h = np.array(h)
    s = np.array(s)
    v = np.array(v)
    r = np.array(r)
    g = np.array(g)
    b = np.array(b)

    length = round(len(b)*0.85)
    h_end = h[length:]
    s_end = s[length:]
    v_end = v[length:]
    b_end = b[length:]
    g_end = g[length:]
    r_end = r[length:]

    for k in range(len(h)):
        if v[k] <= 180:
            blki += 1
    
    for v in range(len(v_end)):
        if v_end[v] <= 180:
            blke += 1


    grays = (b.astype(int) + g.astype(int) + r.astype(int))/3  # Compute grayscale with naive equation
    
    #debugging
    # if round(angle) == 93:
    #     cv2.line(tester, (circ_col, circ_row), (col,row), (255,0,0), thickness=1)
    # if round(angle) == 180:
    #     cv2.line(tester, (circ_col, circ_row), (col,row), (255,0,0), thickness=1)
    
    #grays = (b.astype(int) + g.astype(int))/2
    #grays_end = (b_end.astype(int) + g_end.astype(int) + r_end.astype(int))/3
    #grays_end = (b_end.astype(int) + g_end.astype(int))/2
    blkis.append(blki)
    blkes.append(blke)
    stds.append(np.std(grays))
    means.append(np.mean(grays))
    gray_values.append(grays)
    # stds_end.append(np.std(grays_end))
    # means_end.append(np.mean(grays_end))
    # gray_values_end.append(grays_end)

df["angles"] = angles
df["stds"] = stds
df["means"] = means
df["gray_values"] = gray_values
df["blkis"] = blkis
df["blkes"] = blkes
# df["stds_end"] = stds_end
# df["means_end"] = means_end
# df["gray_values_end"] = gray_values_end

min_mean = df["means"].min()
#max_std_end = df["stds_end"].max()
angle_end = df.loc[df["means"] == min_mean, "angles"].values[0]
print(angle_end)
(pt_col, pt_row) = df.loc[df["means"] == min_mean, "indices"].values[0]
imcopy = image.copy()
cv2.line(imcopy, (circ_col, circ_row), (pt_col, pt_row), (0, 255, 0), thickness=3)  # Draw needle radial line
#(pt_col, pt_row) = df.loc[df["stds_end"] == max_std_end, "indices"].values[0]
#cv2.line(imcopy, (circ_col, circ_row), (pt_col, pt_row), (255, 0, 0), thickness=3)
cv2.circle(imcopy, (circ_col, circ_row), circ_rad, (0, 0, 255), thickness=3)

print("Process finished --- %s seconds ---" % (time.time() - start_time))  
#Plot mean pixel value as a function of needle "clock angle" (zero degrees is 12 o'clock)
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax2.scatter(df["angles"], df["stds"], color="r", label="pixel std. dev.", alpha=0.3)
ax.scatter(df["angles"], df["means"], label="pixel mean", color="b", alpha=0.3)
ax.scatter(df["angles"], df["blkis"], label="black normal", color="g", alpha=0.3)
ax.scatter(df["angles"], df["blkes"], label="black end", color="y", alpha=0.3)
# ax.scatter(df["angles"], df["means_end"], label="pixel mean end", color="g", alpha=0.3)
# ax.scatter(df["angles"], df["stds_end"], label="pixel std. dev end", color="y", alpha=0.3)
ax2.legend(loc="lower center")
ax.legend(loc="lower left")
ax.set_xlabel("angle")
ax.set_ylabel("Metric Value along Radial Line")
ax.set_title("Locating Gauge Needle from Radial Line Pixel Values", fontsize=16)



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

cv2.imshow('lines',lines)
cv2.waitKey(0)

cv2.imshow('test',tester)
cv2.waitKey(0)

cv2.imshow("result",imcopy)
cv2.waitKey(0)

plt.show()

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