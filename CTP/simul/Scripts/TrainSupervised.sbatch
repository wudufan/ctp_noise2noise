#!/bin/bash

#SBATCH -n 3
#SBATCH -w gpu-node006
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J n2cCTP

for beta in 12.5 25 37.5 50
do
./TrainSupervised.sh 0 $beta 100000 &
./TrainSupervised.sh 2 $beta 200000 &
./TrainSupervised.sh 3 $beta 1e+06 &
wait
done




