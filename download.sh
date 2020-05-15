#!/bin/bash

if [ $# -eq 0 ]; then
  DATA_DIR="/tmp"
else
  DATA_DIR="$1"
fi

# Install required packages
python3 -m pip install -r requirements.txt

# Get TF Lite model and labels
curl -O https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip

unzip mobilenet_v1_1.0_224_quant_and_labels.zip -d ${DATA_DIR}

rm mobilenet_v1_1.0_224_quant_and_labels.zip

pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl