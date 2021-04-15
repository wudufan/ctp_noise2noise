#!/bin/bash

DEVICE=$1
BETA=$2
N0=${3:-100000}

NAME=supervised_beta_${BETA}_N0_${N0}
echo $NAME

cd ..

python3 TrainSupervised.py --device $DEVICE --imgshapeIn 256 256 1 --imgshapeOut 256 256 1 --beta $BETA \
--imgFile /home/dwu/trainData/Noise2Noise/train/ctp/simul/data/imgs_${N0}.npy \
--outDir /home/dwu/trainData/Noise2Noise/train/ctp/simul/$NAME > Outputs/${NAME}.txt

