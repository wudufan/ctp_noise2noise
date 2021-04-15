#!/bin/bash

#SBATCH -n 1
#SBATCH -w gpu-node006
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J n2cCTP

for beta in 0 12.5 25 37.5 50
do
./TestSupervised.sh 1 $beta
done




