#!/bin/bash

#SBATCH -n 3
#SBATCH -w gpu-node006
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J n2nCTP

for N0 in 200000 1e+06 100000
do
./TrainFrameToAvg.sh 0 0 $N0 &
./TrainFrameToAvg.sh 2 25 $N0 &
./TrainFrameToAvg.sh 3 50 $N0 &
wait

./TrainFrameToAvg.sh 2 12.5 $N0 &
./TrainFrameToAvg.sh 3 37.5 $N0 &
wait
done

