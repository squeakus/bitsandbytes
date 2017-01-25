import os,sys,re

ppmList = []
for fileName in os.listdir(os.getcwd()):
    if fileName.endswith('.ppm'):
        ppmList.append(fileName)

def get_file_num(filename): return float(re.findall(r'\d+',filename)[0])
def numsort(a,b): return cmp(get_file_num(a),get_file_num(b))
 
ppmList.sort(lambda x,y:cmp(get_file_num(x),get_file_num(y)))


for ppmName in ppmList:
    print ppmName
