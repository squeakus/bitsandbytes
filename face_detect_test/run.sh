#!/bin/bash

# Base Script File (run.sh)
# Created: Sun 23 Feb 2020 22:20:48 GMT
# Version: 1.0
#
# This Bash script was developped by François-Xavier Thomas.
# You are free to copy, adapt or modify it.

python3 face_detection_test.py -m_fd ~/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml -i 0 -d_fd MYRIAD
