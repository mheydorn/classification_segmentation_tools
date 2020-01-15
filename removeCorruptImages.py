import os 
import glob
import cv2
badCount = 0
for f in glob.glob("ValidationData/*.png"):
    img = cv2.imread(f, 1)
    if img is None:
        badCount += 1
        os.system("rm " + f)

print( badCount)
