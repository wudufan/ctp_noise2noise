#!/bin/bash

DEVICE=$1
BETA=$2

NAME=beta_${BETA}
echo $NAME

cd ..

python3 TrainFrameToAvg.py --device $DEVICE --imgshapeIn 256 256 1 --imgshapeOut 256 256 1 --beta $BETA --lr 1e-4 \
--outDir /home/dwu/trainData/Noise2Noise/train/ctp/real/$NAME > Outputs/${NAME}.txt

