#!/bin/bash

#SBATCH -n 4
#SBATCH -w node002
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J GaussianCTP

for std in 0 1 2 3 4
do
    ./Gaussian.sh $std &
done
wait




