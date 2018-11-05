#!/bin/bash

python3 track_object.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --video input/cat.mp4 --label cat --output output/cat_output.avi
