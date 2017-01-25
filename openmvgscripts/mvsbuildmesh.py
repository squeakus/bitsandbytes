#!/usr/bin/python
# Build mesh

# Indicate the openMVG binary directory

import commands
import os
import subprocess
import sys

# Indicate the openMVG camera sensor width directory

def main(folder, global_recon=True):
    os.chdir(folder)
    run_cmd("openMVG_main_openMVG2openMVS -i sfm_data.bin - o scene.mvs")
    run_cmd("DensifyPointCloud scene.mvs")
    run_cmd("ReconstructMesh scene_dense.mvs")
    run_cmd("RefineMesh scene_dense_mesh.mvs")
    run_cmd("TextureMesh --export-type obj scene_dense_mesh_refine.mvs")

def run_cmd(cmd):
    print cmd
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE)
    result = process.communicate()
    return result

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ("Usage %s <global_results_dir>" % sys.argv[0])
        sys.exit(1)
    main(sys.argv[1])
