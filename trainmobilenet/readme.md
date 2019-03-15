# How to train mobilenetSSD

This code is based on two tutorials:
- Caffe: https://tolotra.com/2018/09/15/how-to-retrain-ssd-mobilenet-for-real-time-object-detection-using-a-raspberry-pi-and-movidius-neural-compute-stick/
- Tensorflow: https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9

# Steps to build:
- Download caffenetssd from my github
- Make sure you are on the ssd branch
- Fix the makefile.config to see HDF5 and the correct opencv libraries
- change the pythonpath to point to the caffe build

# Data labelling:
- use labelImg to create xml labels
- delete any images without labels
- create the following file structure
MyDatasetName/
        |_Images/
        |_Labels/
        |_Structure/ 
- add labelmap.prototxt to root:
item {
  name: "none_of_the_above"
  label: 0
  display_name: "background"
}
item {
  name: "person"
  label: 1
  display_name: "person"
} 
- call createtrainvals on the dataset to create the trainvals
- copy trainval.txt to test.txt ?!?
- use the create_data.sh script in data/localData to create the training vals
- clone chuanqi305's mobilenet version to the ./examples folder
..- git clone --depth 1 https://github.com/chuanqi305/MobileNet-SSD

- create symlinks to the prototxt  and lmdbs
..- ln -s ../../raccoons/labelmap.prototxt .
..- ln -s ../../raccoons/raccoons/lmdb/raccoons_trainval_lmdb trainval_lmdb
..- ln -s ../../raccoons/raccoons/lmdb/raccoons_test_lmdb test_lmdb
- call gen_model with the number of classes (background counts as a class): ./gen_model.sh 2
