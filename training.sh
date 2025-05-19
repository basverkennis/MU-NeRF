#!/bin/bash
#SBATCH -p GPUExtended
#SBATCH -N 1
#SBATCH -t 0-96:00
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err
#SBATCH --gres=gpu:1

if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/usr/local/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/usr/local/anaconda3/bin:$PATH"
fi

source activate TestEnvCondaGPU4EDU

cd /home/u869018/ActiveNeRF-GPU4EDU/Original-Test-GPUEDU\ Complete/ActiveNeRF

# Train ActiveNeRF
python -u run_nerf.py --config configs/complete.txt --basedir "./logs" --init_image 3 --active_iter 40000 80000 120000 160000 --i_all 200000 --choose_k 3 --N_rand 4096 --N_samples 64 --N_importance 128 --chunk 65536 --netchunk 262144 --factor 4 --i_print 1000 --i_weights 1000

# Train MU-NeRF
python -u run_nerf_MU.py --config configs/complete.txt --basedir "./logs" --init_image 3 --active_iter 40000 80000 120000 160000 --i_all 200000 --choose_k 3 --N_rand 4096 --N_samples 64 --N_importance 128 --chunk 65536 --netchunk 262144 --factor 4 --i_print 1000 --i_weights 1000