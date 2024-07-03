#!/bin/bash
#SBATCH -J flow_training
#SBATCH -o %x.out
#SBATCH -e %x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --mail-type=END
#SBATCH --mail-user=mmdesai@mit.edu

## python environment
source /home/software.ppc64le/spack/v0.16.2/spack/opt/spack/linux-rhel8-power9le/gcc-8.3.1/anaconda3-2020.02-2ks5tchtak3kzzbryjloiqhusujnh67c/etc/profile.d/conda.sh
conda activate /home/mmdesai/.conda/envs/my_torch
python flow_training.py
