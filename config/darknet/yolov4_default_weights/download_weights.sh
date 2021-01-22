#!/bin/bash

if test $1 -eq 0; then
    echo "Weights of yolov4 will not be downloaded."
elif [ -f $2/yolov4.weights ]; then
    echo "Weights of yolov4 are already downloaded."
else
    if test $1 -eq 1; then
        echo "Downloading Weights"
        echo "Downloading yolov4.weights"
        wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -4  -P $2
    fi
fi
