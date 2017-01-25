import os, sys, subprocess
mypath = os.getcwd()
filelist = os.listdir(mypath)

def run_cmd(cmd):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    result = process.communicate()
    return result


for filename in filelist:
    
    if filename.endswith('.jpg'):
        oldname = filename
        oldname = oldname.replace('(','\(')
        oldname = oldname.replace(')','\)')
        filename = filename.rstrip('.jpg')
        filename = filename.replace('(','.')
        filename = filename.replace(')','-')
        namelist = filename.split('_')

        trackname = namelist[1]
        timestamp = namelist[2]+'-'+namelist[3]
        newname = timestamp+trackname+'.jpg'
        print "oldname", oldname,"new name:", newname
        run_cmd("mv "+oldname+" "+newname)
