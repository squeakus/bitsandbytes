#!/bin/bash
while true ; do python3 train.py --train_dir training --pipeline_config_path training/pipeline.config --logtostderr; echo "Will restart training in 60s..."; sleep 60; done

