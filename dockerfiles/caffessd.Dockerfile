#FROM ubuntu:16.04

# add /usr/include/hdf5/serial/ to the include folder in the makefile.config
FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04
WORKDIR /workspace
RUN apt update && apt install -y aptitude cmake git cmake-curses-gui vim build-essential libatlas-base-dev libgflags-dev libgoogle-glog-dev liblmdb-dev libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler python-pip libopenblas-dev python-numpy python-skimage ipython
RUN apt install -y --no-install-recommends libboost-all-dev
RUN pip install protobuf


RUN ln -s /usr/lib/x86_64-linux-gnu/libhdf5_serial_hl.so.10 /usr/lib/x86_64-linux-gnu/libhdf5_hl.so
RUN ln -s /usr/lib/x86_64-linux-gnu/libhdf5_serial.so.10 /usr/lib/x86_64-linux-gnu/libhdf5.so
RUN git clone git clone https://user:pass@gitlab.devtools.intel.com/ccheron/caffe-ssd.git
WORKDIR /workspace/caffe
RUN git checkout ssd
RUN cp Makefile.config.example Makefile.config
RUN echo "INCLUDE_DIRS := \$(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/" >> Makefile.config
RUN make -j4
RUN make py
#RUN make test -j4
RUN export PYTHONPATH=$PYTHONPATH:/workspace/caffe/python
