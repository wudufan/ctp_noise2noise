#!/bin/bash

#SBATCH -n 3
#SBATCH -w node002
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J GaussianCTP

for std in 0 0.5 1 2 3 4 5
do
    ./Gaussian.sh $std 100000 &
    ./Gaussian.sh $std 200000 &
    ./Gaussian.sh $std 1e+06 &
    wait
done




