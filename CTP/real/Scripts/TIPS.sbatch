#!/bin/bash

#SBATCH -n 4
#SBATCH -w gpu-node006
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J TIPSCTP

for std2 in 1 2 3 4
do
    ./TIPS.sh 0 0.025 $std2 &
    ./TIPS.sh 0 0.05 $std2 &
    ./TIPS.sh 0 0.075 $std2 &
    ./TIPS.sh 0 0.1 $std2 &
    wait
done




