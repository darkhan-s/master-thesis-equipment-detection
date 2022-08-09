#!/bin/bash
## in slurm, ## starts a line with comments (not executed)

#SBATCH --job-name=thesis_quickTest
#SBATCH --account=project_2005695
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
##SBATCH --mail-type=ALL
##SBATCH --output=slurm-detectron2-adapt-output_%j.txt
##SBATCH --error=slurm-detectron2-adapt-errors_%j.txt
#SBATCH --mem 8G 

### For testing: 
#SBATCH --partition=gputest
#SBATCH --time=00:15:00


ml purge
##ml pytorch/1.10

## env variables for debug
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

## in the future detectron2 should be used without utilizing the conda environment as it puts unnecessary stress on slurm and 
## running multiple training loops becomes not possible.
## here, detectron2 etc. are installed to conda because of facing problems when trying to install 
## compatible versions of pytorch cuda etc directly to csc slurm 

### install these if missing ###
##conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 opencv -c pytorch -c conda-forge
##python -m pip install detectron2 --user -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
source /projappl/project_2005695/miniconda3/etc/profile.d/conda.sh
conda activate base

## remove older test outputs 
rm -rf ./output-mymodel-quickTest


############## FINAL FULL TRAINING (quick tests to validate inputs) ############### 
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #   --dataset_path "/scratch/project_2005695/PyTorch-CycleGAN/datasets/" \
# #   --mode 2 \
# #   --config configs/faster_rcnn_R101_cross_tless.yaml \
# #   OUTPUT_DIR "./output-mymodel-quickTest" \
# #   SOLVER.MAX_ITER 3000 \
# #   DATALOADER.DEBUG_LIMIT_INPUT 300 \
# #   INPUT.CROP.ENABLED True \
# #   SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.0 \
# #   SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.0 \
# #   SEMISUPNET.DIS_LOSS_WEIGHT 0.05


############## FINAL RESUME TRAINING (quick tests to validate inputs) ############### 
# # retrain with my components added and augmentation for the class 21 based on model trained on 1-20
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --resume \
# #     --dataset_path "/scratch/project_2005695/PyTorch-CycleGAN/datasets/" \
# #     --mode 2 \
# #     --config configs/faster_rcnn_R101_cross_tless_incremental_21.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     OUTPUT_DIR "./output-mymodel-quickTest" \
# #     SOLVER.MAX_ITER 3000 \
# #     DATALOADER.DEBUG_LIMIT_INPUT 300 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True \
# #     MODEL.WEIGHTS output-mymodel-classes-1-20-FINAL-MyModel_withCustomAugmentation/model_best_wo_solver_states.pth



## for training with pumps (all processes are the same, but change the dataset parameters)
## srun python train_net_extended_class.py --num-gpus 1 --dataset_path "/scratch/project_2005695/master-thesis-equipment-detection/bin/pumps/" --mode 1 --config configs/faster_rcnn_R101_cross_pump.yaml OUTPUT_DIR "./output-mymodel-quickTest"  DATALOADER.DEBUG_LIMIT_INPUT 300 


## For interactive slurm commands on cpu (to validate inputs), run for example: 
## sinteractive --account project_2005695 --time 00:15:00 --cores 6 python train_net_extended_class.py --dist-url tcp://127.0.0.1:53385 --config configs/faster_rcnn_R101_cross_tless_full.yaml OUTPUT_DIR "./output-mymodel-quickTest" MODEL.DEVICE cpu DATALOADER.DEBUG_LIMIT_INPUT 300
