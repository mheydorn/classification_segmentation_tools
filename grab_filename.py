from pathlib import Path
import os
from IPython import embed

file1 = open("rizochtania","r") 
lines =  file1.readlines() 


lines2 = []
for l in lines:
    lines2.append(l.strip())

lines = lines2
os.system("rm got/*")
count = 0
for filename in Path('./segmented').rglob('*.png'):
    #print(filename)
    filename = str(filename)
    basename = os.path.basename(filename)
    print (basename)
    if basename in lines:
        count += 1
        print("got one")
        os.system("cp -l " + filename + " got")

print(count)
