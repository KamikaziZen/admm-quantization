#!/bin/bash
#SBATCH --job-name=EVALUATE
#SBATCH --partition gpu_devel
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:0
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=d.cherniuk@skoltech.ru
#SBATCH --output=logs/%x.%j.log

module load gpu/cuda-11.3
module load python/anaconda3


CUDA_HOME=/trinity/shared/opt/cuda-11.3
CUDA_PATH=/trinity/shared/opt/cuda-11.3
source activate mark20
python scripts/evaluate.py --layer=$1 --rank=$2