#!/bin/bash

BETA1=$1
BETA2=$2
N0=${3:-100000}

NAME=TV_beta_${BETA1}_${BETA2}_N0_${N0}
echo $NAME

cd ..

python3 TV.py --betas $BETA1 $BETA1 $BETA2 \
--imgFile /home/dwu/trainData/Noise2Noise/train/ctp/simul/data/imgs_${N0}.npy \
--outFile /home/dwu/trainData/Noise2Noise/train/ctp/simul/tv/${NAME}.npz > Outputs/TV/${NAME}.txt

