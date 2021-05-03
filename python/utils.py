import os
import subprocess
import math
import glob


def main():
    print("a utility class of static functions")


def yes_or_no(question):
    while "the answer is invalid":
        reply = str(raw_input(question + " (y/n): ")).lower().strip()
        if reply[0] == "y":
            return True
        if reply[0] == "n":
            return False


def cartesian2polar(x, y):
    r = math.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(x, y)
    xback = r * math.cos(theta)
    yback = r * math.sin(theta)
    return r, theta


def normalize(x):
    normed = (x - min(x)) / (max(x) - min(x))
    return normed


def run_cmd(cmd):
    print(cmd)
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE
    )
    result = process.communicate()
    return result


def check_cvsdk():
    # check the environment is set
    if "INTEL_CVSDK_DIR" in os.environ.keys():
        ov_dir = os.environ["INTEL_CVSDK_DIR"]
        print("The openvino path is", ov_dir)
    else:
        print("please intialise OpenVINO environment,run setupvars!")
        exit()
    return ov_dir


def create_folder(foldername):
    if os.path.exists(foldername):
        print("folder already exists:", foldername)
    else:
        os.makedirs(foldername)


def find(regex, folder="./"):
    found = []
    for filename in glob.iglob(folder + "/**/" + regex, recursive=True):
        found.append(filename)
    return found


def ave(values):
    return float(sum(values)) / len(values)


def std(values, ave):
    return math.sqrt(float(sum((value - ave) ** 2 for value in values)) / len(values))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print(f"usage: python {sys.argv[0]} <foldername>")

if __name__ == "__main__":
    main()
