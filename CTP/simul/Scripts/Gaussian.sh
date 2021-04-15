#!/bin/bash

STD=$1
N0=${2:-100000}

NAME=Gaussian_std_${STD}_N0_${N0}
echo $NAME

cd ..

python3 Gaussian.py --std $STD \
--imgFile /home/dwu/trainData/Noise2Noise/train/ctp/simul/data/imgs_${N0}.npy \
--outFile /home/dwu/trainData/Noise2Noise/train/ctp/simul/gaussian/${NAME} > Outputs/Gaussian/${NAME}.txt

