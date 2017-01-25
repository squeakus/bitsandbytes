import sys,os,re, subprocess, time

popFolder=os.getcwd()+"/population/"
meshList=[]
ppmList=[]

def createPPM(fileName):
    print "creating ppms"
    cmd = 'ffmedit -xv 600 600 '+popFolder+fileName
    print cmd
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    process.communicate()

for fileName in os.listdir(popFolder):
            if fileName.endswith('.ppm'):
                ppmList.append(fileName)
            elif fileName.endswith('.mesh'):
                meshList.append(fileName)

for mesh in meshList:
    createPPM(mesh)

cmd = "convert "+popFolder+"gen*.ppm run.gif"
print cmd
process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
process.communicate()
print "finished creating gif"


