#!/usr/bin/env sh
export PYTHONPATH=$PYTHONPATH:/home/passwd123/work/mtcnn-caffe-master/64net
set -e
/home/passwd123/work/caffe/build/tools/caffe train \
	 --solver=./solver.prototxt \
  	 #--weights=./48net-only-cls.caffemodel
