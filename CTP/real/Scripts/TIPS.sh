#!/bin/bash

DEVICE=$1
STD1=$2
STD2=$3

NAME=TIPS_sigma_${STD1}_${STD2}
echo $NAME

cd ..

python3 TIPS.py --device $DEVICE --stdTips $STD1 --stdDist $STD2 \
--outFile /home/dwu/trainData/Noise2Noise/train/ctp/real/tips/${NAME} > Outputs/TIPS/${NAME}.txt

