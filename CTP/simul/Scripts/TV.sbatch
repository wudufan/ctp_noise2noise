#!/bin/bash

#SBATCH -n 3
#SBATCH -w gpu-node006
#SBATCH -c 8
#SBATCH -p defq
#SBATCH -J TVCTPSimul

for beta1 in 1e-3 7.5e-4 5e-4 2.5e-4 1e-4
do
    for beta2 in 1e-3 7.5e-4 5e-4 2.5e-4 1e-4 7.5e-5
    do
        ./TV.sh $beta1 $beta2 100000 &
        ./TV.sh $beta1 $beta2 200000 &
        ./TV.sh $beta1 $beta2 1e+06 &
        wait
    done
done




