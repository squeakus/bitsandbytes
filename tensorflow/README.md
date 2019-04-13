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
- Convert to openvino IR
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
 export PYTHONPATH=$PYTHONPATH:<research_folder>:<research_folder>/slim
 ```

3. if you are using python 3 you may get a [unicode error](https://stackoverflow.com/questions/19877306/nameerror-global-name-unicode-is-not-defined-in-python-3), replace any unicode cast with str e.g.; unicode(blah) -> str(blah)
4. Download a pretrained copy of the network you want to use from [the model detection zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
 - I would recommend mobilenetV2_coco as it runs very quickly on the NCS
 - faster RCNN can also run on openvino but it takes a long time to load the model
5. Copy the folder containing the network to models/research/object_detection
6. Now that tensorflow object detection code is set up, follow one of the tutorials above on labelling your dataset and converting it to the tfrecord format
7. If you have a folder of labelled data you can use voc2tfrecord.py in this repo to automatically generate the tfrecords and the pbtxt file. 
8. Once you have converted the dataset copy it to the object detector folder.
9. We are going to use the train.py and eval.py in the legacy folder to execute the network. Copy them from the legacy folder to the object_detection folder
10. Train the network with the following command:
```bash
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/pipelinev2.config
```
11. Check the performance of the network with eval.py:
 ```bash
 python eval.py --logtostderr --pipeline_config_path=pipelinev2.config --checkpoint_dir=training/ --eval_dir=eval/
 ```
12. View the results using tensorboard: 
```bash
#To visualize the eval results
tensorboard --logdir=eval/
#TO visualize the training results
tensorboard --logdir=training/
```
13. Once training is complete, freeze the graph with the following command:
```bash
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/pipelinev2.config --trained_checkpoint_prefix training/model.ckpt-6227 --output_directory training/frozen
```

## Conversion steps:
1. Convert the frozen graph to fp16 intermediate format using the config pipeline:
```bash
$INTEL_CVSDK_DIR/deployment_tools/model_optimizer/mo_tf.py --input_model=frozen_inference_graph.pb --tensorflow_use_custom_operations_config $INTEL_CVSDK_DIR/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --output_dir FP16 --data_type FP16
```

3. use openvino_ssd_image.py and openvino_ssd_video.py scripts from this repo to execute the IR on the NCS.
