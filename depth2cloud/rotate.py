#!/usr/bin/python
import sys, subprocess
from numpy import arange

def main():
    if not sys.argv[1].endswith('.bin'):
        print "rotate <binfile>"
        exit()
    name = sys.argv[1]
    print  "opening", name

    plystr = "CloudCompare "
    for i in arange(0,360,45):
        print i
        run_cmd("python rotatecloud.py "+ name +" "+ str(i))
        plyname = name.rstrip('.bin')+str(i)+".ply"
        plystr += plyname + " "
    print plystr
    run_cmd(plystr)

def run_cmd(cmd, debug = False):
    """execute commandline command cleanly"""
    if debug:
        print cmd
    else:
        cmd += " > /dev/null 2>&1"
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE)
    result = process.communicate()
    return result

if __name__ == '__main__':
    main()
