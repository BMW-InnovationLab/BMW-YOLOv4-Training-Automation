#!/bin/bash

echo "Downloading weights..."
config/darknet/yolo_default_weights/download_weights.sh 1 "config/darknet/yolo_default_weights"
config/darknet/yolov4_default_weights/download_weights.sh 1 "config/darknet/yolov4_default_weights"

docker build -t $USER/yolov4:latest -f docker/Dockerfile .
