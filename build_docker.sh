#!/bin/bash

docker build -t $USER/yolov4:latest -f docker/Dockerfile --build-arg DOWNLOAD_ALL=1 .
