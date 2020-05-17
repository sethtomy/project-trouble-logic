#!/bin/bash

sudo apt-get install -y libopenjp2-7 libtiff5 python3-numpy

python3 -m pip install -r requirements.txt

pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl