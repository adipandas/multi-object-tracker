#!/bin/sh

# MobileNet-SSD model reference https://github.com/chuanqi305/MobileNet-SSD/

wget --no-check-certificate "https://drive.google.com/u/0/uc?id=0B3gersZ2cHIxRm5PMWRoTkdHdHc&export=download" -O 'MobileNetSSD_deploy.caffemodel'
wget "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/daef68a6c2f5fbb8c88404266aa28180646d17e0/MobileNetSSD_deploy.prototxt" -O "MobileNetSSD_deploy.prototxt"
