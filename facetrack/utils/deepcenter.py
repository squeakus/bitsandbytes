# import necessary packages
import os
import imutils
import cv2
import pantilthat as pth
from openvino.inference_engine import IENetwork, IEPlugin


class ObjCenter:
    def __init__(self, model_xml):
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)

        net = IENetwork(model=model_xml, weights=model_bin)
        input_blob = next(iter(net.inputs))
        out_blob = next(iter(net.outputs))
        self.exec_net = plugin.load(network=net, num_requests=2)

        self.n, self.c, self.h, self.w = net.inputs[input_blob].shape
        self.cur_request_id = 0
        self.next_request_id = 1
        del net


    def update(self, frame, frameCenter):
        initial_h, initial_w, depth = frame.shape
        in_frame = cv2.resize(next_frame, (self.w, self.h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((self.n, self.c, self.h, self.w))
        self.exec_net.start_async(request_id=next_request_id, inputs={input_blob: in_frame})

        rects = []
        if self.exec_net.requests[cur_request_id].wait(-1) == 0:
            inf_end = time.time()
            det_time = inf_end - inf_start

            # Parse detection results of the current request
            res = self.exec_net.requests[cur_request_id].outputs[out_blob]
            for obj in res[0][0]:
                # Draw only objects when probability more than specified threshold
                if obj[2] > args.prob_threshold:
                    xmin = int(obj[3] * initial_w)
                    ymin = int(obj[4] * initial_h)
                    xmax = int(obj[5] * initial_w)
                    ymax = int(obj[6] * initial_h)
                    rects.append([xmin, ymin, xmax - xmin, ymax - ymin]) 

        # check to see if a face was found
        if len(rects) > 0:
            # extract the bounding box coordinates of the face and
            # use the coordinates to determine the center of the
            # face
            (x, y, w, h) = rects[0]
            faceX = int(x + (w / 2))
            faceY = int(y + (h / 2))

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
        pth.clear()
        pth.show()
        return (frameCenter, None)