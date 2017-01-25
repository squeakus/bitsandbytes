import nxt, thread, time, subprocess, timeit, sys

def main():
    if len(sys.argv) < 2:
        print "usage: python rotate2.py foldername"
        exit()

    foldername = sys.argv[1]
    start = timeit.default_timer()
    print "Looking for brick"
    b = nxt.find_one_brick()
    print "found it"
    l = nxt.Motor(b, nxt.PORT_B)
    r = nxt.Motor(b, nxt.PORT_C)
    m = nxt.SynchronizedMotors(l, r, 0)

    for i in range(100):
        print "image:",i
        m.turn(80,75)
        time.sleep(2)
        take_img()

    stop = timeit.default_timer()
    print "time taken:", stop - start
    execute_command("mkdir " + foldername)
    execute_command("mv *.jpg " + foldername)
    

def take_img():
    execute_command('gphoto2 --capture-image-and-download --force-overwrite --filename "img%Y-%m-%d-%H-%M-%S.jpg"', False)
    time.sleep(3)

def execute_command(cmd, wait=True):
   process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                              stdin=subprocess.PIPE)
   if wait:
       process.communicate()

if __name__ == '__main__': main()
