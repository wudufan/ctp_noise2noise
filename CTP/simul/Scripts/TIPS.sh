#!/bin/bash

DEVICE=$1
STD1=$2
STD2=$3
N0=${4:-100000}

NAME=TIPS_sigma_${STD1}_${STD2}_N0_${N0}
echo $NAME

cd ..

python3 TIPS.py --device $DEVICE --stdTips $STD1 --stdDist $STD2 \
--imgFile /home/dwu/trainData/Noise2Noise/train/ctp/simul/data/imgs_${N0}.npy \
--outFile /home/dwu/trainData/Noise2Noise/train/ctp/simul/tips/${NAME}.npz > Outputs/TIPS/${NAME}.txt

