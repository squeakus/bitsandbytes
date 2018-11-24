FROM ubuntu:18.04
# Set up all the pre requisites
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Ireland
WORKDIR /workspace
RUN apt update && apt install -y wget cmake git cmake-curses-gui vim build-essential libatlas-base-dev libgflags-dev libgoogle-glog-dev liblmdb-dev libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler python-pip
RUN apt install -y --no-install-recommends libboost-all-dev
RUN pip install numpy protobuf

# build special branch of caffe
RUN git clone https://github.com/BVLC/caffe.git
WORKDIR /workspace/caffe
RUN git remote add yosinski https://github.com/yosinski/caffe.git
RUN git fetch --all
RUN git checkout --track -b deconv-deep-vis-toolbox yosinski/deconv-deep-vis-toolbox
WORKDIR /workspace/caffe/build
RUN cmake ../ -DCPU_ONLY=ON
RUN make -j4
RUN make install
ENV PYTHONPATH="/workspace/caffe/python:$PYTHONPATH"

# build deepviz
RUN apt install -y python-opencv python-scipy python-skimage
WORKDIR /workspace
RUN git clone https://github.com/yosinski/deep-visualization-toolbox
WORKDIR /workspace/deep-visualization-toolbox
RUN cp models/caffenet-yos/settings_local.template-caffenet-yos.py settings_local.py
WORKDIR /workspace/deep-visualization-toolbox/models/caffenet-yos/
RUN ./fetch.sh
WORKDIR /workspace/deep-visualization-toolbox
RUN echo "caffevis_caffe_root='/workspace/caffe/python'\ncaffevis_mode_gpu = False" >> settings_local.py
