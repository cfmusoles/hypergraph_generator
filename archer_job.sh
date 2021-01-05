#!/bin/bash --login

# name of the job
#PBS -N job  
# how many nodes
#PBS -l select=1
# walltime
#PBS -l walltime=4:00:0
# budget code
#PBS -A e582

# This shifts to the directory that you submitted the job from
cd $PBS_O_WORKDIR

module load python-compute/3.6.0_gcc6.1.0


# bandwidth matrix creation

aprun -n 1 python3 symmetric_hgraph_generator.py

