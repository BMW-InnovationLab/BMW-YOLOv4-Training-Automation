#!/bin/bash -x

if test $1 -eq 0; then
    echo "Weights will not be downloaded"
else
    if test $1 -eq 1; then
        echo "Downloading Weights"
        echo "Downloading yolov3.weights"
        wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights -P $2 -4
        echo "Downloading yolov3-tiny.weights"
        wget https://pjreddie.com/media/files/yolov3-tiny.weights -P $2 -4 
        echo "Downloading darknet53.conv.74"
        wget https://pjreddie.com/media/files/darknet53.conv.74 -P $2 -4 
        echo "All weights downloaded!"
    fi
fi
