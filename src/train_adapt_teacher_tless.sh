#!/bin/bash
## in slurm, ## starts a line with comments (not executed)

#SBATCH --job-name=thesisFinal
#SBATCH --account=project_2005695
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=ALL
##SBATCH --output=slurm-detectron2-adapt-output_%j.txt
##SBATCH --error=slurm-detectron2-adapt-errors_%j.txt
#SBATCH --mem 64G 

### For training (up to 36 hours): 
#SBATCH --partition=gpusmall
#SBATCH --time=12:00:00

### For short tasks: 
##SBATCH --partition=gputest
##SBATCH --time=00:15:00


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


#################################################################################################
### FINAL experiment TRAINING 
## train without any custom components 
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --dataset_path "/scratch/project_2005695/PyTorch-CycleGAN/datasets/" \
# #     --mode 2 \
# #     --config configs/faster_rcnn_R101_cross_tless_full.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "WarmupTwoStageMultiStepLR"\
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     OUTPUT_DIR "./output-mymodel-classes-1-30-FINAL-Original_withoutMyComponents" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.0 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.0 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS False 

# # ## train without custom components but with a new LR scheduler 
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --dataset_path "/scratch/project_2005695/PyTorch-CycleGAN/datasets/" \
# #     --mode 2 \
# #     --config configs/faster_rcnn_R101_cross_tless_full.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     OUTPUT_DIR "./output-mymodel-classes-1-30-FINAL-Original_newScheduler" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.0 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.0 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS False 
# # ## train with custom augmentation 
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --dataset_path "/scratch/project_2005695/PyTorch-CycleGAN/datasets/" \
# #     --mode 2 \
# #     --config configs/faster_rcnn_R101_cross_tless_full.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     OUTPUT_DIR "./output-mymodel-classes-1-30-FINAL-withCustomAugmentation" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.0 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.0 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True 
## train with my components added and augmentation
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --dataset_path "/scratch/project_2005695/PyTorch-CycleGAN/datasets/" \
# #     --mode 2 \
# #     --config configs/faster_rcnn_R101_cross_tless_full.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     OUTPUT_DIR "./output-mymodel-classes-1-30-FINAL-MyModel_withCustomAugmentation-1" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True 

## train with my components added and augmentation but without the new scheduler 
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --dataset_path "/scratch/project_2005695/PyTorch-CycleGAN/datasets/" \
# #     --mode 2 \
# #     --config configs/faster_rcnn_R101_cross_tless_full.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "WarmupTwoStageMultiStepLR" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     OUTPUT_DIR "./output-mymodel-classes-1-30-FINAL-MyModel_withCustomAugmentation-origScheduler" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True 
## train with my components added and augmentation for first 20 classes only
# # ## train with my components added and augmentation for first 20 classes only
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --config configs/faster_rcnn_R101_cross_tless.yaml \
# #   --dataset_path "/scratch/project_2005695/PyTorch-CycleGAN/datasets/" \
# #   --mode 2 \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     OUTPUT_DIR "./output-mymodel-classes-1-20-FINAL-MyModel_withCustomAugmentation" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True 
# # ## train with my components added and augmentation for the class 21 only
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --dataset_path "/scratch/project_2005695/PyTorch-CycleGAN/datasets/" \
# #     --mode 2 \
# #     --config configs/faster_rcnn_R101_cross_tless_one_class.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     OUTPUT_DIR "./output-mymodel-classes-21only-FINAL-MyModel_withCustomAugmentation" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True 


# # have to have the weights to run this 
# # Remove the optimizer states etc. from the weights
# srun python trim_detectron_model.py \
#     --pretrained_path output-mymodel-classes-1-20-FINAL-MyModel_withCustomAugmentation/model_best.pth   \
#     --save_path output-mymodel-classes-1-20-FINAL-MyModel_withCustomAugmentation \
#     --config_file configs/faster_rcnn_R101_cross_tless.yaml 

# # retrain with my components added and augmentation for the class 21 based on model trained on 1-20
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --resume \
# #     --dataset_path "/scratch/project_2005695/PyTorch-CycleGAN/datasets/" \
# #     --mode 2 \
# #     --config configs/faster_rcnn_R101_cross_tless_incremental_21.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     OUTPUT_DIR "./output-mymodel-classes-1-20to1-20and21-FINAL-MyModel_withCustomAugmentation" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True \
# #     MODEL.WEIGHTS output-mymodel-classes-1-20-FINAL-MyModel_withCustomAugmentation/model_best_wo_solver_states.pth


## train my custom model with original scheduler on pump images
srun python train_net_extended_class.py \
    --num-gpus 1 \
    --dataset_path "/scratch/project_2005695/master-thesis-equipment-detection/bin/pumps/" \
    --mode 1 \
    --config configs/faster_rcnn_R101_cross_pump.yaml \
    SOLVER.LR_SCHEDULER_NAME "WarmupTwoStageMultiStepLR" \
    SOLVER.BASE_LR 0.001 \
    SOLVER.MAX_ITER 50000 \
    SOLVER.WEIGHT_DECAY 0.001 \
    OUTPUT_DIR "./output-mymodel-pumps-FINAL-MyModel_withCustomAugmentation" \
    SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
    SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
    SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
    SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
    INPUT.CROP.ENABLED False \
    INPUT.CUSTOM_AUGMENTATIONS True 

##################################################################
