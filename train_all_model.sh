#!/bin/bash

python train.py --model mlp --head linear
python train.py --model resnet --head linear
python train.py --model vgg --head linear

python train.py --model mlp --head arcface
python train.py --model resnet --head arcface
python train.py --model vgg --head arcface