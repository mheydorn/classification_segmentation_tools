import os 
import glob
from IPython import embed
import random

#'clump=0.26 good=0.02 green=0.00 growth_crack=0.01 misshapen=0.50 new_bruise=0.01 no_netting_good=0.00 old_bruise=0.20 rub=0.00 -2019-09-25-18-34-01.668531.png.png'
list_output = "list_output"
allFiles = glob.glob("sorted_by_thresh/*")

classes = ["clump", "good", "green", "growth_crack", "misshapen", "new_bruise", "no_netting_good", "old_bruise", "rub"]

for c in classes:
    write_file = open(list_output + "/" + c + ".txt","w")
    entries = {}
    for i, f in enumerate(glob.glob("sorted_by_thresh/*")):
        base = os.path.basename(f)
        find_string = c + "="
        start = base.find(find_string)
        start = start + len(find_string)
        end = start + 4
        score = float(base[start:end])
        if entries.get(score) == None:
            entries[score] = []
        entries[score].append(base)

    for k in sorted(entries.keys()):
        for ff in entries[k]:
            write_file.write(ff + "\n")
    write_file.close()
        
        
        

        




    
    
    
    
