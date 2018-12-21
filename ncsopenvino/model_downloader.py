import subprocess
import sys

def main():
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
              'dilation',
              'googlenet-v1',
              'googlenet-v2',
              'googlenet-v4',
              'alexnet',
              'ssd_mobilenet_v2_coco',
              'resnet-50',
              'resnet-101',
              'resnet-152',
              'googlenet-v3']

    for model in models:
        cmd = "python3 downloader.py --name " + model
        run_cmd(cmd)


def run_cmd(cmd):
    print(cmd)
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE)
    result = process.communicate()
    print(result)
    return result

if __name__ == "__main__":
    main()
