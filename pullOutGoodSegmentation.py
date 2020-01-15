import cv2
import glob
import numpy as np
import os
from IPython import embed

for i, f in enumerate(glob.glob("/data/tater_pipeline/classification/Dataset_Sep_04/green/*")):
    img = cv2.imread(f, 0)
    total_non_zero = np.count_nonzero(img)
    total_pixels = img.shape[0] * img.shape[1]
    goodToBadRatio = total_non_zero / float(total_pixels)

    if total_pixels < 20000:
        continue

    #blur = cv2.blur(gray, (3, 3)) # blur the image

    img = cv2.blur(img, (3, 3)) # blur the image

    img[img > 0] = 1
    thresh = img
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    # create hull array for convex hull points
    hull = []
     
    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], False))

    # create an empty black image
    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
     
    hullRatios = []
    #if len(contours) != 1:
    #    print("problem")
    #    exit()
    # draw contours and hull points
    for i in range(len(contours)):
        color_contours = (0, 255, 0) # green - color for contours
        color = (255, 0, 0) # blue - color for convex hull
        # draw ith contour
        cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        # draw ith convex hull object
        cv2.drawContours(drawing, hull, i, color, 1, 8)

        hullArea = cv2.contourArea(hull[0])
        contourArea = cv2.contourArea(contours[0])
        hullRatios.append(float(contourArea) / float(hullArea))


    r = min(hullRatios)

    print (r)
    #cv2.imshow("draw", drawing)
    #cv2.waitKey(0)


    #print (i)
    basename = os.path.basename(f)
    if r > 0.96:
        
        os.system("cp -l " + f + " /data/tater_pipeline/classification/Dataset_Sep_04/good_segmentation/")
    else:
        os.system("cp -l " + f + " /data/tater_pipeline/classification/Dataset_Sep_04/bad_segmentation")
    
