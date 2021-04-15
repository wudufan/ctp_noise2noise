#!/bin/bash

#SBATCH -n 3
#SBATCH -w gpu-node004
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J n2nCTP

./TrainFrameToAvg.sh 1 0 &
./TrainFrameToAvg.sh 2 25 &
./TrainFrameToAvg.sh 3 50 &
wait

./TrainFrameToAvg.sh 2 12.5 &
./TrainFrameToAvg.sh 3 37.5 &
wait

