#!/bin/bash

sudo apt-get update
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    wget \
    jq \
    gnupg-agent \
    software-properties-common
