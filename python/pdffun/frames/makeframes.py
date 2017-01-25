import os, subprocess
savelist =[]

def run_cmd(cmd):
    print cmd
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE)
    result = process.communicate()
    return result



for fileName in os.listdir('.'):
    if fileName.endswith('.pdf'):
        savelist.append(fileName)

for pdf in savelist:
    newname = 'frame'
    newname += pdf.rstrip('.pdf')
    newname += '.gif'
    run_cmd('rm -f *.png')
    run_cmd('convert '+pdf+' -background white +matte out%03d.png')
    run_cmd('montage *.png -tile 4x '+newname)
    
    
