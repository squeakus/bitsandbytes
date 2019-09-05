# import necessary packages
import os
import imutils
import cv2
import pantilthat as pth
from openvino.inference_engine import IENetwork, IEPlugin



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
        
        # center of object
        self.midx = 0
        self.midy = 0
        # center of next face found
        self.center_x = 0
        self.center_y = 0
        # dimensions of rectangle for face
        self.x = 0
        self.y = 0
        self.rw = 0
        self.rh = 0
        

    def update(self, next_frame, frameCenter):
        initial_h, initial_w, depth = next_frame.shape
        in_frame = cv2.resize(next_frame, (self.w, self.h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((self.n, self.c, self.h, self.w))
        self.exec_net.start_async(request_id=self.next_request_id, inputs={self.input_blob: in_frame})

        rects = []
        # comparative distance to find face closest to center of frame
        # out of group of faces before any face is found
        prev_distance = 1000
        
        if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
            # Parse detection results of the current request
            res = self.exec_net.requests[self.cur_request_id].outputs[self.out_blob]

            faces = []
            
            # for each object in our results
            for obj in res[0][0]:
                # Draw only faces when probability more than specified threshold
                # if object is a face, add to list of faces
                if obj[2] > 0.5:
                    faces.append(obj)
                
            # if we have faces
            if len(faces) > 0:
                # for each object in our list of faces, do calculations for center
                # we know each object is a face and do not need to compare to probability again
                for obj in faces:
                    # bottom left corner
                    xmin = int(obj[3] * initial_w)
                    ymin = int(obj[4] * initial_h)
                    # top right corner
                    xmax = int(obj[5] * initial_w)
                    ymax = int(obj[6] * initial_h)
                                        
                    # center of object
                    mid_x = (xmax + xmin)/2
                    mid_y = (ymax + ymin)/2
                    
                    # distance from center of new object to previous center
                    distance = (((self.midx - mid_x)**2)+((self.midy - mid_y)**2))**(1/2)
                    
                    # find object center closest to previous center
                    # set new values to compare against next object
                    if distance < prev_distance:
                        prev_distance = distance

                        self.center_x = mid_x
                        self.center_y = mid_y

                        self.x = xmin
                        self.y = ymin
                        self.rw = xmax - xmin
                        self.rh = ymax - ymin
                
                # take final object (closest to previous object)
                # (re)set values
                self.midx = self.center_x
                self.midy = self.center_y
                prev_distance = 1000
                
                # add rectangle for the face found
                rects.append([self.x, self.y, self.rw, self.rh])

        self.cur_request_id, self.next_request_id = self.next_request_id, self.cur_request_id
        # check to see if a face was found
        if len(rects) > 0:
            # reassign name to object's center
            faceX = self.midx
            faceY = self.midy

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

            # if there is a face, return the face center and rectangle
            return ((faceX, faceY), rects[0])

        
        pth.clear()
        pth.show()
        
        # if there is no face, return the frame center and no rectangle (to stay still)
        return (frameCenter, None)

