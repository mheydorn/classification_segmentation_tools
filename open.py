import cv2
import glob
from IPython import embed
import matplotlib.pyplot as plt

for f in glob.glob("/data/tater_pipeline/segmentation/masks_2018.Aug.14/*"):
    img = cv2.imread(f, 1)
    plt.imshow(img)
    plt.show()
    embed()
