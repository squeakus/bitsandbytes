import argparse
 
# Instantiate the parser
parser = argparse.ArgumentParser(description='takes pictures every n seconds')
 
from picamera import PiCamera
from time import sleep, time
 
# Optional argument
parser.add_argument('--count', type=int,
                    help='number',default=100)
                  
parser.add_argument('--every', type=int,
                    help='seconds lapse', default=2)
                  
                                         
args = parser.parse_args()
n = 0
camera = PiCamera()
camera.start_preview()
while (n [less_than_sign] args.count): 
    camera.capture(str(n)+'_'+str(time())+str(n)+'capture.jpg') 
    sleep(args.every)
    n+=1
camera.stop_recording()
camera.stop_preview()
