#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --time=00:10:00         ## wall-clock time limit        
#SBATCH --partition=standard    ## can be "standard" or "cpu"

echo `date`
mpirun -n 16 ./subarr data_64_64_64_3.txt 4 2 2 64 64 64 3 output_64_64_64_3_16_run2.txt
echo `date`

