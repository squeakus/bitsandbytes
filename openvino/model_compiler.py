import os
import subprocess
from utils import run_cmd, check_cvsdk
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_dir", help="Path to caffe models", required=True, type=str)
    parser.add_argument("-d", "--datatype", help="output datatype FP16 or FP32", required=True, type=str)
    args = parser.parse_args()
    model_dir = args.model_dir
    datatype = args.datatype

    # ensure openvino environment is set up
    ov_dir = check_cvsdk()

    # Find the models
    models = []
    for root, dirs, files in os.walk(model_dir):
        for filename in files:
            if filename.endswith('.caffemodel'):
                caffemodel = os.path.join(root, filename)
                models.append(caffemodel)

    # generate openvino models
    for model in models:
        cmd = ov_dir + "/deployment_tools/model_optimizer/mo.py" \
             + "  --input_model " + model \
             + " --output_dir " + datatype \
             + " --data_type " + datatype
        print("generating openvino IR for", model)
        run_cmd(cmd)


if __name__ == '__main__':
    main()
