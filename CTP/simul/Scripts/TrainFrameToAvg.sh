#!/bin/bash

DEVICE=$1
BETA=$2
N0=${3:-100000}

NAME=beta_${BETA}_N0_${N0}
echo $NAME

cd ..

python3 TrainFrameToAvg.py --device $DEVICE --imgshapeIn 256 256 1 --imgshapeOut 256 256 1 --beta $BETA --lr 1e-4 \
--imgFile /home/dwu/trainData/Noise2Noise/train/ctp/simul/data/imgs_${N0}.npy \
--outDir /home/dwu/trainData/Noise2Noise/train/ctp/simul/$NAME > Outputs/${NAME}.txt

