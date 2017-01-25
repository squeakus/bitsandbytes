import subprocess
import platform
import sys

sys.path.append("/home/jonathan/data/linuxapps/caffe/python")
import caffe
caffe.set_mode_gpu()
import lmdb

from sklearn.cross_validation import StratifiedShuffleSplit
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

print "OS:     ", platform.platform()
print "Python: ", sys.version.split("\n")[0]
print "CUDA:   ", subprocess.Popen(["nvcc","--version"], stdout=subprocess.PIPE).communicate()[0].split("\n")[3]
print "LMDB:   ", ".".join([str(i) for i in lmdb.version()])

net = caffe.Net("model_prod.prototxt","./_iter_100001.caffemodel", caffe.TEST)
l, f = get_data_for_case_from_lmdb("./test_data_lmdb/", "00001230")
out = net.forward(**{net.inputs[0]: np.asarray([f])})

# if the index of the largest element matches the integer
# label we stored for that case - then the prediction is right
print np.argmax(out["prob"][0]) == l, "\n", out
plt.bar(range(9),out["prob"][0])


from google.protobuf import text_format
from caffe.draw import get_pydot_graph
from caffe.proto import caffe_pb2
from IPython.display import display, Image

_net = caffe_pb2.NetParameter()
f = open("model_prod.prototxt")
text_format.Merge(f.read(), _net)
display(Image(get_pydot_graph(_net,"TB").create_png()))

# weights connecting the input with relu1
arr = net.params["ip1"][0].data
In [222]:
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
fig.colorbar(cax, orientation="horizontal")
cax = ax.matshow(arr, interpolation='none')

_ = plt.hist(arr.tolist(), bins=20)
