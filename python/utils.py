import os
import subprocess
import math
import glob

def main():
    print('a utility class of static functions')

def cartesian2polar(x, y):
    r = math.sqrt(x**2 + y**2)
    theta = np.arctan2(x, y)
    xback = r * math.cos(theta);
    yback = r * math.sin(theta);
    return r, theta

def run_cmd(cmd):
    print(cmd)
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE)
    result = process.communicate()
    return result

def check_cvsdk():
    # check the environment is set
    if 'INTEL_CVSDK_DIR' in os.environ.keys():
        ov_dir = os.environ['INTEL_CVSDK_DIR']
        print("The openvino path is", ov_dir)
    else:
        print("please intialise OpenVINO environment,run setupvars!")
        exit()
    return ov_dir

def create_folder(foldername):
    if os.path.exists(foldername):
        print('folder already exists:', foldername)
    else:
        os.makedirs(foldername)

def find(regex, folder='./'):
    found = []
    for filename in glob.iglob(folder+'**/'+regex, recursive=True):
        found.append(filename)
    return found

def ave(values):
    return float(sum(values)) / len(values)

def std(values, ave):
    return math.sqrt(float(sum((value - ave) ** 2
                               for value in values)) / len(values))

if __name__ == "__main__":
    main()
