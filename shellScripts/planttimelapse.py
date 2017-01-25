import subprocess, time

def main():
    counter = 0
    countfile = open('counter.txt', 'r')
    counter = int(countfile.readline())
    print "starting at image:", counter
    countfile.close()
    
    while True:
        filename = "img%05d.jpg" % counter 
        print "saving image", filename,"at", time.strftime("%H:%M:%S")
        run_cmd("gphoto2 --capture-image-and-download --force-overwrite")
        time.sleep(10)
        run_cmd("mv capt0000.jpg "+filename) 
        time.sleep(600)
        counter += 1
        countfile = open('counter.txt','w')
        countfile.write(str(counter))
        countfile.close()

def run_cmd(cmd, debug = False):
    """execute commandline command cleanly"""
    if debug:
        print cmd
    else:
        cmd += " > /dev/null 2>&1"
        process = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stdin=subprocess.PIPE)
    result = process.communicate()
    return result


if __name__=='__main__':
    main()
