import subprocess
import os
from utils import run_cmd

def main():
    # check the environment is set
    if 'INTEL_CVSDK_DIR' in os.environ.keys():
        ov_dir = os.environ['INTEL_CVSDK_DIR']
        print("The openvino path is", ov_dir)
    else:
        print("please intialise the OpenVINO environment,run setupvars!")
        exit()

    models = ['densenet-121',
              'densenet-161',
              'densenet-169',
              'densenet-201',
              'squeezenet1.0',
              'squeezenet1.1',
              'mtcnn-p',
              'mtcnn-r',
              'mtcnn-o',
              'mobilenet-ssd',
              'vgg19',
              'vgg16',
              'ssd512',
              'ssd300',
              'inception-resnet-v2',
              'googlenet-v1',
              'googlenet-v2',
              'googlenet-v4',
              'alexnet',
              'resnet-50',
              'resnet-101',
              'resnet-152',
              'googlenet-v3']

    download = ov_dir + "/deployment_tools/model_downloader/downloader.py --name "
    for model in models:
        cmd = download + model
        run_cmd(cmd)


if __name__ == "__main__":
    main()
