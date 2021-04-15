#!/bin/bash

#SBATCH -n 3
#SBATCH -w gpu-node006
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J TIPSCTP

for std1 in 0.025 0.05 0.075 0.1 0.25
do
    for std2 in 1 2 3 4 5 7.5
    do
        ./TIPS.sh 1 $std1 $std2 100000 &
        ./TIPS.sh 2 $std1 $std2 200000 &
        ./TIPS.sh 3 $std1 $std2 1e+06 &
        wait
    done
done




