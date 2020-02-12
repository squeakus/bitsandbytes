#!/bin/bash

# Base Script File (runcpu.sh)
# Created: Wed 12 Feb 2020 21:33:57 GMT
# Version: 1.0
# Author: jonathan byrne jonathanbyrn@gmail.com>
#
# This Bash script was developped by Jonathan Byrne
# You are free to copy, adapt or modify it.
./gaze_estimation_demo -i cam\
 -m ~/intel/openvino_2020.1.023/deployment_tools/open_model_zoo/tools/downloader/intel/gaze-estimation-adas-0002/FP32-INT8/gaze-estimation-adas-0002.xml\
 -m_hp ~/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/intel/head-pose-estimation-adas-0001/FP32-INT8/head-pose-estimation-adas-0001.xml\
 -m_fd ~/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/intel/face-detection-adas-0001/FP32-INT8/face-detection-adas-0001.xml\
 -m_lm ~/intel/openvino_2020.1.023/deployment_tools/open_model_zoo/tools/downloader/intel/facial-landmarks-35-adas-0002/FP32-INT8/facial-landmarks-35-adas-0002.xml
