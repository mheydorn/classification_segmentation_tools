import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import glob
from IPython import embed

for f in glob.glob("/data/tater_sai_no_sai/test/label/*.mat"):
    img = loadmat(f)
    embed()


