# Tensorflow object detectors to NCS library
The aim of these scripts is take data you have labelled and train a tensorflow graph to run on the Neural compute stick. It is based on the this detailed tutorial [tensorflow object detection tutorial](https://becominghuman.ai/tensorflow-object-detection-api-tutorial-training-and-evaluating-custom-object-detector-ed2594afcf73)

# Overview
- Use [labelImg](https://github.com/tzutalin/labelImg) to label your images
- Install the [tensorflow models repository](https://github.com/tensorflow/models)

- Split the images into train and test datasets either manually or using [this script](https://github.com/squeakus/bitsandbytes/blob/master/tensorflow/voc2tfrecord.py)
- Download a pretrained model
- Edit the pipeline.config file for your dataset
- Retrain the model
- Freeze the network
- Convert to openvino Intermediate Representation (IR)
- Run IR on the stick


If you want to try and do this on a windows machine (may god have mercy on your soul), you can follow [this tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)

## Training Steps:
1. Download the [tensorflow models repository](https://github.com/tensorflow/models)
2. Follow the instructions for installing the [object detector libraries](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
3. Once you have installed the dependencies you will need to compile the protobufs
 ```bash
 # From tensorflow/models/research/'
 protoc object_detection/protos/*.proto --python_out=.
 python3.6 setup.py build
 python3 setup.py install
 # Add the command below to your .bashrc file to permanently add TF to your pythonpath
 export PYTHONPATH=$PYTHONPATH:<Your Home>/models/tensorflow/research/:<Your Home>/models/tensorflow/research/slim

 ```

4. if you are using python 3 you may get a [unicode error](https://stackoverflow.com/questions/19877306/nameerror-global-name-unicode-is-not-defined-in-python-3), replace any unicode cast with str e.g.; unicode(blah) -> str(blah)
5. Download a pretrained copy of the network you want to use from [the model detection zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
 - I would recommend [mobilenetV2_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz) as it runs very quickly on the NCS
 - [faster RCNN](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz) can also run on openvino but it takes a long time to load the model
6. Copy the folder containing the network to models/research/object_detection
7. Now that tensorflow object detection code is set up, follow one of the tutorials above on labelling your dataset and converting it to the tfrecord format
8. Once you have a folder of images and labels you can use the voc2tfrecord.py script in this repo to automatically generate the tfrecords and the pbtxt file.
9. Take the output of the converted dataset (tfrecords, pbtxt) and copy it to the object detector folder.
10. We are going to use the train.py and eval.py in the legacy folder to execute the network. Copy them from the legacy folder to the object_detection folder
11. Train the network with the following command:
```bash
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/pipelinev2.config
```
12. Check the performance of the network with eval.py:
 ```bash
 python eval.py --logtostderr --pipeline_config_path=training/pipelinev2.config --checkpoint_dir=training/ --eval_dir=eval/
 ```
  If you get unicode error - Please change ```unicode``` to ```str```
13. View the results using tensorboard:
```bash
#To visualize the eval results
tensorboard --logdir=eval/
#TO visualize the training results
tensorboard --logdir=training/
```
14. Once training is complete, freeze the graph with the following command:
```bash
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/pipelinev2.config --trained_checkpoint_prefix training/model.ckpt-6227 --output_directory training/frozen
```

## Conversion steps:
1. Convert the frozen graph to fp16 intermediate format using the config pipeline:
```bash
# For mobilenet
$INTEL_CVSDK_DIR/deployment_tools/model_optimizer/mo_tf.py --input_model=frozen_inference_graph.pb --tensorflow_use_custom_operations_config $INTEL_CVSDK_DIR/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --output_dir FP16 --data_type FP16

# For faster RCNN
python3 $INTEL_CVSDK_DIR/deployment_tools/model_optimizer/mo.py --framework tf --input_model frozen_inference_graph.pb --tensorflow_use_custom_operations_config $INTEL_CVSDK_DIR/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config --data_type FP16
```

3. For mobilenet use openvino_ssd_image.py and openvino_ssd_video.py scripts from this repo to execute the IR on the NCS.

4. For Faster RCNN, you need to compile the openvino samples:
	- Run the inference_engine/samples/buildsamples.sh
	- Copy object_detection_sample_ssd from inferenceenginebuild/intel/release folder to the folder with your IR.
```bash
./object_detection_sample_ssd -i ~/models_share/img/480x270/office/CES_FD_FR_High_view_original_01904.bmp -m ~/models_share/frcnn/resnet50/faster_rcnn_resnet50_coco_2018_01_28/vpu/frozen_inference_graph.xml -pc -d MYRIAD
```
