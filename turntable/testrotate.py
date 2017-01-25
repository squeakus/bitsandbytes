import nxt, thread, time, subprocess, timeit

def main():

    start = timeit.default_timer()
    print "Looking for brick"
    b = nxt.find_one_brick()
    print "found it"
    l = nxt.Motor(b, nxt.PORT_B)
    r = nxt.Motor(b, nxt.PORT_C)
    m = nxt.SynchronizedMotors(l, r, 0)

    for i in range(10):
        print "image:",i
        m.turn(80,3600)
        #take_img()
        time.sleep(2)

    stop = timeit.default_timer()
    print "time taken:", stop - start


def take_img():
    execute_command('gphoto2 --capture-image-and-download --force-overwrite --filename "img%Y-%m-%d-%H-%M-%S.jpg"', False)
    time.sleep(5)

def execute_command(cmd, wait=True):
   process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                              stdin=subprocess.PIPE)
   if wait:
       process.communicate()


if __name__ == '__main__': main()
