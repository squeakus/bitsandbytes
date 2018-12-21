import os
import subprocess

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
                print(caffemodel)
                models.append(caffemodel)



    for model in models:
        cmd = ov_dir + "/deployment_tools/model_optimizer/mo.py" \
             + "  --input_model " + model \
             + " --output_dir " + datatype \
             + " --data_type " + datatype

        run_cmd(cmd)


def run_cmd(cmd):
    print(cmd)
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE)
    result = process.communicate()
    print(result)
    return result

if __name__ == '__main__':
    main()