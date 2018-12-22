import os
import subprocess
from utils import run_cmd, check_cvsdk

def main():
    model_dir = "./classification"
    datatype = "FP16"

    # ensure openvino environment is set up
    check_cvsdk()

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
