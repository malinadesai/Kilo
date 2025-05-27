#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:1
#SBATCH -C centos7
#SBATCH -p sched_mit_hill 
#SBATCH --mem-per-cpu=4000
#SBATCH -o output_%j.txt
#SBATCH -e error_%j.txt 
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=mmdesai@mit.edu


source ~/.bashrc
conda activate nmma_enviro

echo "ACTIVATED ENVIRONMENT"

lightcurve-generation --model Ka2017 \
	--outdir test_lc_sup \
	--label test_50 \
	--tmin -2 --tmax 20 --dt 0.25 --filters ztfg,ztfr,ztfi --outfile-type json \
	--injection test_lc_sup/injection_test_var_50.json \
	--injection-detection-limit 22.0,22.0,22.0 --ztf-uncertainties --increment-seeds 
