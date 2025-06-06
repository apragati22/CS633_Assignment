#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --time=00:10:00         ## wall-clock time limit        
#SBATCH --partition=standard    ## can be "standard" or "cpu"

echo `date`
mpirun -n 8 ./src data_64_64_96_7.bin.txt 2 2 2 64 64 96 7 output_64_64_96_7_8_run2.txt
echo `date`

