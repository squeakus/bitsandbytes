# import necessary packages
import os
import imutils
import cv2
import pantilthat as pth
from openvino.inference_engine import IENetwork, IEPlugin

#import colorsys
from utils.pid import PID
#import math
import time



class ObjCenter:
    def __init__(self, model_xml):
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        plugin = IEPlugin(device="MYRIAD")

        net = IENetwork(model=model_xml, weights=model_bin)
        self.input_blob = next(iter(net.inputs))
        self.out_blob = next(iter(net.outputs))
        self.exec_net = plugin.load(network=net, num_requests=2)

        self.n, self.c, self.h, self.w = net.inputs[self.input_blob].shape
        self.cur_request_id = 0
        self.next_request_id = 1
        del net
        #self.prev_face = [0,0]
        self.midx = 0
        self.midy = 0
        self.center_x = 0
        self.center_y = 0
        self.x = 0
        self.y = 0
        self.rw = 0
        self.rh = 0
        

    def update(self, next_frame, frameCenter):
        initial_h, initial_w, depth = next_frame.shape
        ##print ("next_frame.shape = ",next_frame.shape)
        in_frame = cv2.resize(next_frame, (self.w, self.h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((self.n, self.c, self.h, self.w))
        self.exec_net.start_async(request_id=self.next_request_id, inputs={self.input_blob: in_frame})

        rects = []
        ##print ("rects is ",rects)
        
        #face_tot = 0
        
        #midpoint_x = initial_w*0.5
        #midpoint_y = initial_h*0.5
        prev_distance = 1000
        
        if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
            # Parse detection results of the current request
            res = self.exec_net.requests[self.cur_request_id].outputs[self.out_blob]
            ##print ("res is ",res)
            #faces = res[0][0]
            #goodfaces = []
            faces = []
            
            # for each object in our results
            for obj in res[0][0]:
                # Draw only faces when probability more than specified threshold
                # if object is a face, add to list of faces
                if obj[2] > 0.5:
                    faces.append(obj)
            
            # if we have no faces
            if len(faces) == 0:
                # do something
                ##print ("no face ")
                #p = PID(0.1, 0.001, 0.002)
                #p.initialize()
                #pth.pan(0)
                #pth.tilt(0)
                #time.sleep(0.2)
                pth.servo_enable(1, False)
                pth.servo_enable(2, False)
                ##time.sleep(1.0)
                #pth.servo_enable(1, True)
                #pth.servo_enable(2, True)
                
            # but if we have faces
            else:
                pth.servo_enable(1, True)
                pth.servo_enable(2, True)
                # for each object in our list of faces, do calculations for center
                # we know each object is a face and do not need to compare to probability again
                for obj in faces:                
                    #face_tot += 1
                    
                    ##print ("obj 2 is ", obj [2])
                    xmin = int(obj[3] * initial_w)
                    ymin = int(obj[4] * initial_h)
                    xmax = int(obj[5] * initial_w)
                    ymax = int(obj[6] * initial_h)
                    
                    ##print ("rects is ",rects)
                    
                    mid_x = (xmax + xmin)/2
                    mid_y = (ymax + ymin)/2
                    
                    distance = (((self.midx - mid_x)**2)+((self.midy - mid_y)**2))**(1/2)
                    
                    if distance < prev_distance:
                        prev_distance = distance
                        #global center_x
                        #global center_y
                        self.center_x = mid_x
                        self.center_y = mid_y
                        #global x
                        #global y
                        #global w
                        #global h
                        self.x = xmin
                        self.y = ymin
                        self.rw = xmax - xmin
                        self.rh = ymax - ymin
                        

             #if distance between centroid and self.prev_face < fixed dist then
             #it is the same face!
             #self.prev_face = centroid
            
            self.midx = self.center_x
            self.midy = self.center_y
            prev_distance = 1000
            
            
            rects.append([self.x, self.y, self.rw, self.rh])

        self.cur_request_id, self.next_request_id = self.next_request_id, self.cur_request_id
        # check to see if a face was found
        if len(rects) > 0:
            # extract the bounding box coordinates of the face and
            # use the coordinates to determine the center of the
            # face
            
            ##faceX = int(x + (w / 2))
            ##faceY = int(y + (h / 2))
            faceX = self.midx
            faceY = self.midy
            
            #i = 0
            #for i < face_tot:
            #    (x, y, w, h) = rects[i]
            #    rect_midx = x + (w / 2)
            #    rect_midy = y + (h / 2)
            #    if rect_midx == faceX and rect_midy == faceY:
            #        corr_rects = rects[i]
                            

            # color the error
            pth.set_all(255,0,0)
            if (faceX - frameCenter[0]) > 10:
                pth.set_pixel(0, 255, 255, 255)
            if (faceX - frameCenter[0]) > 30:
                pth.set_pixel(1, 255, 255, 255)
            if (faceX - frameCenter[0]) > 50:
                pth.set_pixel(2, 255, 255, 255)
            if (faceX - frameCenter[0]) < -10:
                pth.set_pixel(7, 255, 255, 255)
            if (faceX - frameCenter[0]) < -30:
                pth.set_pixel(6, 255, 255, 255)
            if (faceX - frameCenter[0]) < -50:
                pth.set_pixel(5, 255, 255, 255)

            pth.show()

            # print("face detected centroid", faceX, faceY)
            # return the center (x, y)-coordinates of the face
            return ((faceX, faceY), rects[0])
 
        # otherwise no faces were found, so return the center of the
        # frame
        
        #if face_tot == 0:
        #   
        #    pth.show()
        #    
        #    p = int(math.sin(time.time()) * 90)
        #    t = int(math.sin(time.time()) * 90)
        #    
        #    pth.pan(p)
        #    pth.tilt(t)
        
        pth.clear()
        pth.show()
        return (frameCenter, None)

