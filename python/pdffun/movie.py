import sys,os,re, subprocess, time

popFolder=os.getcwd()+"/population/"
os.chdir(popFolder)
meshList=[]
ppmList=[]
commandList=[]
commandList.append("ls | grep .jpg > frames.txt")
commandList.append("mencoder mf://@frames.txt -mf w=800:h=600:fps=10:type=jpg -ovc xvid -ovc x264 -x264encopts bitrate=3000:pass=1:nr=2000 -o a.avi")
commandList.append("rm -f *.ppm")
commandList.append("rm -f *.jpg")


def createPPM(fileName):
    print "creating ppms"
    if sys.platform == 'linux2':
        cmd = './linuxMedit '+popFolder+fileName
    else:
        cmd = 'ffmedit -xv 600 600 '+popFolder+fileName
    print cmd
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    process.communicate()

for fileName in os.listdir(popFolder):    
    if fileName.endswith('.mesh'):
        meshList.append(fileName)

for mesh in meshList:
    createPPM(mesh)

for fileName in os.listdir(popFolder):
    if fileName.endswith('.ppm'):
        print "found ppm"
        ppmList.append(fileName)

for ppmFile in ppmList:
    jpgFile=ppmFile.replace('ppm','jpg')
    cmd = "convert "+popFolder+ppmFile+" "+popFolder+jpgFile
    print cmd
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    process.communicate()

for cmd in commandList:
    print cmd
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    process.communicate()
