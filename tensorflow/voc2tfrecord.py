"""
Create tfrecords from VOC datasets, automatically partiition into train and test records
This assumes the images are in an image folder and the labels are in annotations
Please update the classes list before running

This is based on the tensorflow tutorial:
https://becominghuman.ai/tensorflow-object-detection-api-tutorial-training-and-evaluating-custom-object-detector-ed2594afcf73
the script does most of the dataset preparation so you can skip to step 3.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import glob
import pathlib
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET
from shutil import copy, rmtree
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
from random import random

CLASSES = ['raccoon']

def main(_):
    test = 0.2 # percentage of data for test
    imagepath = "/home/jonathan/data/raccoons/images"
    labelpath = "/home/jonathan/data/raccoons/annotations"

    train_test_split(imagepath, labelpath, test)

    xml_df = xml_to_csv("./train")
    xml_df.to_csv('./train/train.csv', index=None)
    xml_df = xml_to_csv("./test")
    xml_df.to_csv('./test/test.csv', index=None)
    print('Successfully converted xml to csv.')

    create_folder("./data")
    create_tfrecord("./train/train.csv", "./data/train.tfrecord", "./train/")
    create_tfrecord("./test/test.csv", "./data/test.tfrecord", "./test/")
    write_pbtxt()

def write_pbtxt():
    outfile = open('object_detection.pbtxt', 'w')
    for idx, item in enumerate(CLASSES):
        outfile.write("item { \n")
        outfile.write("  id: " + str(idx+1) + "\n")
        outfile.write("  name:" + item + "\n")
        outfile.write("}\n")

def create_tfrecord(csvfile, outdir, imagedir):
    writer = tf.python_io.TFRecordWriter(outdir)
    path = os.path.join(imagedir)
    examples = pd.read_csv(csvfile)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(outdir)
    print('Successfully created the TFRecords: {}'.format(output_path))

def create_folder(foldername):
    if os.path.exists(foldername):
        print('folder already exists, recreating folder:', foldername)
        rmtree(foldername)
    
    os.makedirs(foldername)


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label in CLASSES:
        return CLASSES.index(row_label) + 1
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def train_test_split(imagepath, labelpath, test):
    create_folder("train")
    create_folder("test")

    train_list = []
    test_list = []
    print("processing" + labelpath + '/*.xml')
    for xml_file in glob.glob(labelpath + '/*.xml'):
        path = pathlib.PurePath(xml_file)
        image_file = os.path.join(imagepath, path.name.replace('xml', 'jpg'))

        if random() > test:
            dst = "./train"
        else:
            dst = "./test"

        copy(image_file, dst)
        copy(xml_file, dst)

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

if __name__ == '__main__':
    tf.app.run()