import time

def main():
    run_cmd('rm -rf capt0000.jpg')

    for i in range(10):
        filename = "img%04d.jpg" % i
        run_cmd('gphoto2 --capture-image-and-download')

        time.sleep(10)

        run_cmd('mv capt0000.jpg')
def run_cmd(cmd):
    print cmd
    #process = subprocess.Popen(cmd, shell=True,
    #                           stdout=subprocess.PIPE,
    #                           stdin=subprocess.PIPE)
    #result = process.communicate()
    #return result

if __name__ == "__main__":
    main()
