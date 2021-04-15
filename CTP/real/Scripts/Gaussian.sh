#!/bin/bash

STD=$1

NAME=Gaussian_std_${STD}
echo $NAME

cd ..

python3 Gaussian.py --std $STD \
--outFile /home/dwu/trainData/Noise2Noise/train/ctp/real/gaussian/${NAME} > Outputs/Gaussian/${NAME}.txt

