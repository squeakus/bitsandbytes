import os, time, subprocess

def run_cmd(cmd):
    print cmd
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE)
    result = process.communicate()
    return result


for i in range(10):
    frame = "%06d.png" % i
    print frame
    time.sleep(3)
    for i in range(1):
        run_cmd('mplayer tv:// -tv device=/dev/video1 -vo png -frames 1')
    run_cmd('mv 00000001.png '+frame)
