#!/usr/bin/python
# Python script to launch OpenMVG and openMVS on folder

# Indicate the openMVG binary directory

import commands
import os
import subprocess
import sys
import time

# Indicate the openMVG camera sensor width directory
CAMERA_SENSOR_WIDTH_DIRECTORY = "/home/jonathan/data/linuxapps/openMVG/src/openMVG/exif/sensor_width_database"

def main(folder, detector):
    start_time = time.time()
    input_dir = os.path.join(os.getcwd(), folder)
    input_dir = input_dir.rstrip('/')
    output_dir = input_dir+"_result"

    if not os.path.exists(output_dir):
      os.mkdir(output_dir)

    matches_dir = os.path.join(output_dir, detector+"_matches")
    camera_file_params = os.path.join(CAMERA_SENSOR_WIDTH_DIRECTORY, "sensor_width_camera_database.txt")

    print "Processing folder: ", input_dir
    print "output: ", output_dir

    if not os.path.exists(matches_dir):
      os.mkdir(matches_dir)

    # Only compute matches if it doesn't already exist
    if not os.path.isfile(os.path.join(matches_dir, 'geometric_matches')):
        match_time = time.time()
        compute_matches(input_dir, matches_dir, camera_file_params, detector)
        print_time("Matching time:", match_time)
    else:
        print "Matches already computed, starting reconstruction."

    #global reconstruction
    global_time = time.time()
    global_reconstruct(matches_dir, output_dir, detector)
    print_time("global time:", global_time)

    #sequential reconstruction
    #sequential_time = time.time()
    #sequential_reconstruct(matches_dir, output_dir, detector)
    #print_time("sequential time:", sequential_time)
    print_time("total time:",start_time)

def compute_matches(input_dir, matches_dir, camera_file_params, detector):
    print "1.Intrinsics analysis"
    #use this when the camera ccd cannot be found
    # cmd = ["openMVG_main_SfMInit_ImageListing","-f 4800","-i", input_dir, "-o",
    #        matches_dir, "-d", camera_file_params, "-c 3"]
    cmd = ["openMVG_main_SfMInit_ImageListing","-i", input_dir, "-o",
           matches_dir, "-d", camera_file_params, "-c 3"]
    run_cmd(cmd)

    print "2. OpenCV Compute features"
    cmd = ["openMVG_main_ComputeFeatures_OpenCV", "-i", matches_dir+"/sfm_data.json",
        "-o", matches_dir, "-m", detector.upper(), "-f" , "1"]
    run_cmd(cmd)

    if detector in ["orb_opencv","brisk_opencv"]:
        print "2. Compute hamming matches"
        cmd = ["openMVG_main_ComputeMatches",  "-i", matches_dir+"/sfm_data.json",
                "-o", matches_dir, "-f", "1", "-n", "BRUTEFORCEHAMMING"]
        run_cmd(cmd)

    else:
        print "2. Compute euclidean (L2) matches"
        cmd = ["openMVG_main_ComputeMatches",  "-i", matches_dir+"/sfm_data.json",
                "-o", matches_dir, "-f", "1", "-n", "ANNL2"]
        run_cmd(cmd)

def global_reconstruct(matches_dir, output_dir, detector):
    #Reconstruction for the global SfM pipeline
    #- global SfM pipeline use matches filtered by the essential matrices
    #- we reuse photometric matches and perform only the essential matrix filer
    print "2. Compute matches (for the global SfM Pipeline)"
    cmd = ["openMVG_main_ComputeMatches",
           "-i", matches_dir+"/sfm_data.json",
           "-o", matches_dir, "-r", "0.8", "-g", "e"]
    run_cmd(cmd)

    reconstruction_dir = os.path.join(output_dir, detector+"_global")
    print "3. Do Global reconstruction"
    cmd = ["openMVG_main_GlobalSfM",
           "-i", matches_dir+"/sfm_data.json",
           "-m", matches_dir,
           "-o", reconstruction_dir]
    run_cmd(cmd)

    print "5. Colorize Structure"
    cmd =  ["openMVG_main_ComputeSfM_DataColor",
            "-i", reconstruction_dir+"/sfm_data.bin",
            "-o", os.path.join(reconstruction_dir,"colorized.ply")]
    run_cmd(cmd)

    print ("4. Structure from Known Poses (robust triangulation)")
    cmd = ["openMVG_main_ComputeStructureFromKnownPoses",
           "-i", reconstruction_dir+"/sfm_data.bin",
           "-m", matches_dir,
           "-o", os.path.join(reconstruction_dir,"robust.ply")]
    run_cmd(cmd)

def sequential_reconstruct(matches_dir, output_dir, detector):
    print "2. Compute matches"
    cmd = ["openMVG_main_ComputeMatches",  "-i", matches_dir+"/sfm_data.json",
           "-o", matches_dir, "-f", "1", "-n", "ANNL2"]
    run_cmd(cmd)

    reconstruction_dir = os.path.join(output_dir,detector+"_sequential")
    print "3. Do Incremental/Sequential reconstruction"
    #set manually the initial pair to avoid the prompt question
    cmd = ["openMVG_main_IncrementalSfM",  "-i", matches_dir+"/sfm_data.json",
           "-m", matches_dir, "-o", reconstruction_dir]
    run_cmd(cmd)

    print "5. Colorize Structure"
    cmd = ["openMVG_main_ComputeSfM_DataColor",
           "-i", reconstruction_dir+"/sfm_data.bin",
           "-o", os.path.join(reconstruction_dir,"colorized.ply")]
    run_cmd(cmd)

    print "4. Structure from Known Poses (robust triangulation)"
    cmd = ["openMVG_main_ComputeStructureFromKnownPoses",
           "-i", reconstruction_dir+"/sfm_data.bin",
           "-m", matches_dir,
           "-o", os.path.join(reconstruction_dir,"robust.ply")]
    run_cmd(cmd)

def print_time(title, timeval):
    seconds = time.time() - timeval
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print title + "%d:%02d:%02d" % (h, m, s)


def run_cmd(cmd):
    #cmd = ' '.join(cmd)
    print ' '.join(cmd)
    #process = subprocess.Popen(cmd, shell=True,
    #                           stdout=subprocess.PIPE,
    #                           stdin=subprocess.PIPE)
    #result = process.communicate()
    #use these settings if you want to see the output
    process = subprocess.Popen(cmd)
    result = process.wait()
    return result

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print ("Usage %s <image_dir> <akaze_opencv|sift_opencv|surf_opencv|orb_opencv>" % sys.argv[0])
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
