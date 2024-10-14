#!/bin/bash
#SBATCH --job-name=EquilibrateChain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=05:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=yw9071@princeton.edu
#SBATCH --constraint=cascade,skylake


module purge
module load intel/2022.2.0
module load intel-mpi/intel/2021.7.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


srun $HOME/.local/bin/lmp_intel -in inputSingleChain.lmp