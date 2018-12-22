import os
import subprocess
from utils import run_cmd

def main():
    model_dir = "./classification"
    datatype = "FP16"
   
    # check the environment is set
    if 'INTEL_CVSDK_DIR' in os.environ.keys():
        ov_dir = os.environ['INTEL_CVSDK_DIR']
        print("The openvino path is", ov_dir)
    else:
        print("please intia OpenVINO environment,run setupvars!")
        exit()

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
