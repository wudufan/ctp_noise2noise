#!/bin/bash

DEVICE=$1
BETA=$2

NAME=supervised_beta_${BETA}
echo $NAME

cd ..

python3 TestNetwork.py --device $DEVICE --imgshapeIn 256 256 1 --imgshapeOut 256 256 1 --beta $BETA \
--checkPoint /home/dwu/trainData/Noise2Noise/train/ctp/simul/supervised_beta_${BETA}_N0_200000/99 \
--outFile /home/dwu/trainData/Noise2Noise/train/ctp/real/supervised/$NAME > Outputs/Supervised/${NAME}.txt

