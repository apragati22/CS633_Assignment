#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --time=00:10:00         ## wall-clock time limit        
#SBATCH --partition=standard    ## can be "standard" or "cpu"

echo `date`
mpirun -n 32 ./src data_64_64_64_3.bin.txt 4 4 2 64 64 64 3 output_64_64_64_3_32_run1.txt
echo `date`

