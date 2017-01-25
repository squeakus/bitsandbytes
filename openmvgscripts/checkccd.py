#!/usr/bin/python
# Python script to launch OpenMVG and openMVS on folder

# Indicate the openMVG binary directory

import commands
import os
import glob
import subprocess
import sys
import time

START_TIME = time.time()
# Indicate the openMVG camera sensor width directory
CAMERA_SENSOR_WIDTH_DIRECTORY = "/home/jonathan/data/linuxapps/openMVG/src/openMVG/exif/sensor_width_database"

def main(folder):
    imagefiles = get_image_files(folder)
    sampleimage = imagefiles[0]
    found = check_ccd_database(sampleimage)

    if found:
        print "happy days"
    else:
        width, height = get_image_size(sampleimage)
        focal = 1.2 * max(width, height)
        print "manually setting focal length with", focal

def get_image_files(folder):
    imagetypes = ['*.jpg', '*.JPG', '*.png', '*.PNG']
    imagefiles = []
    for files in imagetypes:
        imagefiles.extend(glob.glob(os.path.join(folder,files)))
    return imagefiles

def get_image_size(imgfile):
    sizes = run_cmd("identify -format \"%w %h\" " + imgfile)
    sizes = sizes[0].split(" ")
    width = int(sizes[0])
    height = int(sizes[1])
    return width, height

def check_ccd_database(imgfile):
    model = run_cmd("exiftool -T -model "+ imgfile)[0].rstrip()
    camera_file_params = os.path.join(CAMERA_SENSOR_WIDTH_DIRECTORY, "sensor_width_camera_database.txt")

    db_file = open(camera_file_params,'r')
    found = False

    for line in db_file:
        db_model = line.split(';')[0]
        print model, db_model
        if model == db_model:
            found = True
            break
            
    return found

def run_cmd(cmd):
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE)
    result = process.communicate()
    return result

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ("Usage %s <image_dir>" % sys.argv[0])
        sys.exit(1)
    main(sys.argv[1])
