import sys, time, subprocess
from merge2cloud import CloudParser
from pylab import *

def main():
    if len(sys.argv) != 2:
        print "grabandmerge.py <filename>"
        exit()
    filename = sys.argv[1]    
    talk_cmd = "espeak"
    run_cmd(talk_cmd + ' "Move to point"',True) 
    time.sleep(5)
    run_cmd(talk_cmd + ' "scanning, please stand still"', True) 
    run_cmd('glgrab '+filename+'top 15')
    run_cmd('glgrab '+filename+'base -10')

    parser = CloudParser([''+filename+'top', ''+filename+'base'])
    parser.parse_image(''+filename+'top.bin')
    parser.parse_image(''+filename+'base.bin')
    parser.calc_ranges()
 
    for idx, cloud in enumerate(parser.clouds):
        parser.clouds[idx] = parser.translate_points(cloud)
        parser.clouds[0] = parser.rotate_points(parser.clouds[0])
    parser.saveclouds()
    
    run_cmd(talk_cmd + ' "aligning clouds"') 
    run_cmd('CloudCompare -SILENT -O '+filename+'top.ply -O '+filename+'base.ply -C_EXPORT_FMT PLY -ICP -FARTHEST_REMOVAL -MERGE_CLOUDS')
    run_cmd(talk_cmd + '"finished"')
        
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

    
if __name__=='__main__':
  main()

