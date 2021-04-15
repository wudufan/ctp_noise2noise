#!/bin/bash

BETA1=$1
BETA2=$2

NAME=TV_beta_${BETA1}_${BETA2}
echo $NAME

cd ..

python3 TV.py --betas $BETA1 $BETA1 $BETA2 \
--outFile /home/dwu/trainData/Noise2Noise/train/ctp/real/tv/${NAME} > Outputs/TV/${NAME}.txt

