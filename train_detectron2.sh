#!/bin/bash
#SBATCH --job-name=detectron2TrainFcos
#SBATCH --account=project_2005695
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-detectron2-fcos-output_%j.txt
#SBATCH --error=slurm-detectron2-fcos-errors_%j.txt
#SBATCH --mem 64G 

### For training: 
##SBATCH --partition=gpusmall
##SBATCH --time=01:30:00

### For testing: 
#SBATCH --partition=gputest
#SBATCH --time=00:15:00


ml purge
##ml pytorch/1.10

source /projappl/project_2005695/miniconda3/etc/profile.d/conda.sh
conda activate base
##conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 opencv -c pytorch -c conda-forge
export CUDA_LAUNCH_BLOCKING=1
##python -m pip install detectron2 --user -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html


### Scripts for fcos training

### For training: 
srun python /scratch/project_2005695/master-thesis-equipment-detection/src/cross_teacher_fcos/fcos_train_net.py --config-file /scratch/project_2005695/master-thesis-equipment-detection/src/cross_teacher_fcos/configs/fcos_R_50_FPN_1x.py  
