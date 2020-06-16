FROM ubuntu:18.04
WORKDIR /workspace
RUN apt update && apt install -y cmake git cmake-curses-gui vim build-essential libatlas-base-dev libgflags-dev libgoogle-glog-dev liblmdb-dev libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler python-pip
RUN apt install -y --no-install-recommends libboost-all-dev
RUN pip install numpy

RUN git clone https://github.com/BVLC/caffe.git
WORKDIR /workspace/caffe/build
RUN cmake ../
RUN make -j4
