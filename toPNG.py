import os
import glob
import cv2
import tensorflow as tf
from IPython import embed
import numpy as np
from scipy.io import savemat, loadmat    
import cv2
import matplotlib.pyplot as plt 


def main():

    '''
    files = []
    with open("final/labels") as fp:
        line = fp.readline()
        files.append(os.path.basename(line).split(".png")[0])
        while line:
           print(line)
           line = fp.readline()
           files.append(os.path.basename(line).split(".png")[0])

    files2 = []
    for f in files:
        if not f == "":
            files2.append("labeling/label/" + f + ".mat")
    files = files2
    '''
    
    files = glob.glob("/auto/shared/client_data/wada/mask_creating/finished/label/*.mat")
    for f in files:
        img = loadmat(f)
        img = img["Layer"]
        mask = np.copy(img)
        
        img = cv2.cvtColor(img.astype(np.float32),cv2.COLOR_GRAY2RGB)
        img =  np.dstack( ( img, np.ones((img.shape[0:2])) ) )

        #img[:,:,3][mask == 1] = 1 
        
        cv2.imwrite("final/labels/" + os.path.basename(f).split(".mat")[0] + ".png", np.array(img).astype(np.float32)*255)
    
    '''
    files = glob.glob("final/labels/*.png")

    for f in files: 
        img = cv2.imread(f, 1)
        #embed()
        os.system("rm " + f)
        cv2.imwrite(f, img)
    '''
    

main()




















