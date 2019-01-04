import os
import subprocess
from argparse import ArgumentParser
from utils import run_cmd, check_cvsdk

def main():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_dir", help="Path to models (FP16)", required=True, type=str)
    parser.add_argument("-l", "--labels", help="Labels file", required=True, type=str)
    parser.add_argument("-r", "--results", help="Path to results", required=True, type=str)
    args = parser.parse_args()
    model_dir = args.model_dir
    label_file = args.labels
    result_file = args.results

    # ensure openvino environment is set up
    check_cvsdk()

    # Find the models
    models = []
    for root, dirs, files in os.walk(model_dir):
        for filename in files:
            if filename.endswith('.xml'):
                caffemodel = os.path.join(root, filename)
                models.append(caffemodel)

    # generate openvino models
    for model in models:

        cmd = "python3 classification_runner.py -m " + model + " -i images/cat.jpg -d MYRIAD " + \
              "--labels " + label_file + " -ni 1000 -nt 1 -r " + result_file
        print("testing network", model)
        run_cmd(cmd)


if __name__ == '__main__':
    main()
