import os 
import glob
from IPython import embed

#'clump=0.26 good=0.02 green=0.00 growth_crack=0.01 misshapen=0.50 new_bruise=0.01 no_netting_good=0.00 old_bruise=0.20 rub=0.00 -2019-09-25-18-34-01.668531.png.png'

allFiles = glob.glob("sorted_by_thresh/*")

for i, f in enumerate(glob.glob("sorted_by_thresh/*")):
    base = os.path.basename(f)
    base = base.replace(' ', '=')
    base = base.split('=')

    good = float(base[3]) + float(base[13]) + float(base[17])
    clump = float(base[1])
    bad = 1.0 - float(good) - float(clump)
    
    basee = os.path.basename(f)
    os.system("cp -l " + "\""+ f + "\"" + " sorted_output/" +"{0:.2f}".format(bad) + "_" + str(i) + "_" + basee.split(" ")[-1].split(".png")[0] + ".png")




    
    
    
    
