#!/bin/bash

#SBATCH -n 4
#SBATCH -w gpu-node004
#SBATCH -c 8
#SBATCH -p defq
#SBATCH -J TVCTP

for beta1 in 7.5e-4 1e-3 2.5e-3 5e-3
do
    for beta2 in 5e-5 7.5e-5 1e-4 2.5e-4
    do
        ./TV.sh $beta1 $beta2 &
    done
    wait
done




