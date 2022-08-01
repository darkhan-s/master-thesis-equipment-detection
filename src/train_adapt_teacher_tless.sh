#!/bin/bash
#SBATCH --job-name=detectron2FINALpart2
#SBATCH --account=project_2005695
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=ALL
##SBATCH --output=slurm-detectron2-adapt-output_%j.txt
##SBATCH --error=slurm-detectron2-adapt-errors_%j.txt
#SBATCH --mem 64G 

### For training: 
#SBATCH --partition=gpusmall
#SBATCH --time=12:00:00

### For testing: 
##SBATCH --partition=gputest
##SBATCH --time=00:15:00


ml purge
##ml pytorch/1.10

## env variables for debug
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

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


# # have to have the weights first 
# # Remove the optimizer states etc. from the weights
# srun python trim_detectron_model.py \
#     --pretrained_path output-mymodel-classes-1-20-FINAL-MyModel_withCustomAugmentation/model_best.pth   \
#     --save_path output-mymodel-classes-1-20-FINAL-MyModel_withCustomAugmentation \
#     --config_file configs/faster_rcnn_R101_cross_tless.yaml 

# # retrain with my components added and augmentation for the class 21 based on model trained on 1-20
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --resume \
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

# # ## retrain with my components added and augmentation for the class 22 based on model trained on 1-20
# # # # have to have the weights first 
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --resume \
# #     --config configs/faster_rcnn_R101_cross_tless_incremental_22.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     OUTPUT_DIR "./output-mymodel-classes-1-20to1-20and22-FINAL-MyModel_withCustomAugmentation" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True \
# #     MODEL.WEIGHTS output-mymodel-classes-1-20-FINAL-MyModel_withCustomAugmentation/model_best_wo_solver_states.pth

# # ## retrain with my components added and augmentation for the class 23 based on model trained on 1-20
# # # # have to have the weights first 
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --resume \
# #     --config configs/faster_rcnn_R101_cross_tless_incremental_23.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     OUTPUT_DIR "./output-mymodel-classes-1-20to1-20and23-FINAL-MyModel_withCustomAugmentation" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True \
# #     MODEL.WEIGHTS output-mymodel-classes-1-20-FINAL-MyModel_withCustomAugmentation/model_best_wo_solver_states.pth



# # # # ## train with my components added and augmentation for the class 22 only
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --config configs/faster_rcnn_R101_cross_tless_one_class_22.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     OUTPUT_DIR "./output-mymodel-classes-22only-FINAL-MyModel_withCustomAugmentation" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True   

# # # # ## train with my components added and augmentation for the class 23 only
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --config configs/faster_rcnn_R101_cross_tless_one_class_23.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     OUTPUT_DIR "./output-mymodel-classes-23only-FINAL-MyModel_withCustomAugmentation" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True  

# # # # ## train with my components added and augmentation for the class 24 only
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --config configs/faster_rcnn_R101_cross_tless_one_class_24.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     OUTPUT_DIR "./output-mymodel-classes-24only-FINAL-MyModel_withCustomAugmentation" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True   


# # # # ## train with my components added and augmentation for the class 25 only
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --config configs/faster_rcnn_R101_cross_tless_one_class_25.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     OUTPUT_DIR "./output-mymodel-classes-25only-FINAL-MyModel_withCustomAugmentation" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True  

# # # # ## train with my components added and augmentation for the class 26 only
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --config configs/faster_rcnn_R101_cross_tless_one_class_26.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     OUTPUT_DIR "./output-mymodel-classes-26only-FINAL-MyModel_withCustomAugmentation" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True  
# # # # # # ## train with my components added and augmentation for the class 27 only
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --config configs/faster_rcnn_R101_cross_tless_one_class_27.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     OUTPUT_DIR "./output-mymodel-classes-27only-FINAL-MyModel_withCustomAugmentation" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True  
# # # # # # ## train with my components added and augmentation for the class 28 only
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --config configs/faster_rcnn_R101_cross_tless_one_class_28.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     OUTPUT_DIR "./output-mymodel-classes-28only-FINAL-MyModel_withCustomAugmentation" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True  
# # # # # # ## train with my components added and augmentation for the class 29 only
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --config configs/faster_rcnn_R101_cross_tless_one_class_29.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     OUTPUT_DIR "./output-mymodel-classes-29only-FINAL-MyModel_withCustomAugmentation" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True  
# # # # # # ## train with my components added and augmentation for the class 30 only
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --config configs/faster_rcnn_R101_cross_tless_one_class_30.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     OUTPUT_DIR "./output-mymodel-classes-30only-FINAL-MyModel_withCustomAugmentation" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True  

## TODO:
## retrain with my components added and augmentation for the class 24 based on model trained on 1-20
# # have to have the weights first 
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --resume \
# #     --config configs/faster_rcnn_R101_cross_tless_incremental_24.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     OUTPUT_DIR "./output-mymodel-classes-1-20to1-20and24-FINAL-MyModel_withCustomAugmentation" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True \
# #     MODEL.WEIGHTS output-mymodel-classes-1-20-FINAL-MyModel_withCustomAugmentation/model_best_wo_solver_states.pth

## retrain with my components added and augmentation for the class 25 based on model trained on 1-20
# # have to have the weights first 
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --resume \
# #     --config configs/faster_rcnn_R101_cross_tless_incremental_25.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     OUTPUT_DIR "./output-mymodel-classes-1-20to1-20and25-FINAL-MyModel_withCustomAugmentation" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True \
# #     MODEL.WEIGHTS output-mymodel-classes-1-20-FINAL-MyModel_withCustomAugmentation/model_best_wo_solver_states.pth


## retrain with my components added and augmentation for the class 26 based on model trained on 1-20
# # have to have the weights first 
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --resume \
# #     --config configs/faster_rcnn_R101_cross_tless_incremental_26.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     OUTPUT_DIR "./output-mymodel-classes-1-20to1-20and26-FINAL-MyModel_withCustomAugmentation" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True \
# #     MODEL.WEIGHTS output-mymodel-classes-1-20-FINAL-MyModel_withCustomAugmentation/model_best_wo_solver_states.pth


# # ## retrain with my components added and augmentation for the class 27 based on model trained on 1-20
# # # # have to have the weights first 
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --resume \
# #     --config configs/faster_rcnn_R101_cross_tless_incremental_27.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     OUTPUT_DIR "./output-mymodel-classes-1-20to1-20and27-FINAL-MyModel_withCustomAugmentation" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True \
# #     MODEL.WEIGHTS output-mymodel-classes-1-20-FINAL-MyModel_withCustomAugmentation/model_best_wo_solver_states.pth

## retrain with my components added and augmentation for the class 28 based on model trained on 1-20
# # have to have the weights first 
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --resume \
# #     --config configs/faster_rcnn_R101_cross_tless_incremental_28.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     OUTPUT_DIR "./output-mymodel-classes-1-20to1-20and28-FINAL-MyModel_withCustomAugmentation" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True \
# #     MODEL.WEIGHTS output-mymodel-classes-1-20-FINAL-MyModel_withCustomAugmentation/model_best_wo_solver_states.pth

# # ## retrain with my components added and augmentation for the class 29 based on model trained on 1-20
# # # # have to have the weights first 
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --resume \
# #     --config configs/faster_rcnn_R101_cross_tless_incremental_29.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     OUTPUT_DIR "./output-mymodel-classes-1-20to1-20and29-FINAL-MyModel_withCustomAugmentation" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True \
# #     MODEL.WEIGHTS output-mymodel-classes-1-20-FINAL-MyModel_withCustomAugmentation/model_best_wo_solver_states.pth

# # ## retrain with my components added and augmentation for the class 30 based on model trained on 1-20
# # # # have to have the weights first 
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #     --resume \
# #     --config configs/faster_rcnn_R101_cross_tless_incremental_30.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     OUTPUT_DIR "./output-mymodel-classes-1-20to1-20and30-FINAL-MyModel_withCustomAugmentation" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True \
# #     MODEL.WEIGHTS output-mymodel-classes-1-20-FINAL-MyModel_withCustomAugmentation/model_best_wo_solver_states.pth

# # have to have the weights first 
# # Remove the optimizer states etc. from the weights
# # srun python trim_detectron_model.py \
# #     --pretrained_path output-mymodel-classes-1-20to1-20and21-FINAL-MyModel_withCustomAugmentation/model_best.pth   \
# #     --save_path output-mymodel-classes-1-20to1-20and21-FINAL-MyModel_withCustomAugmentation \
# #     --config_file configs/faster_rcnn_R101_cross_tless.yaml 


###this is to check whether it drops significantly or not
## retrain with my components added and augmentation for the class 22 based on model trained on 1-21
## have to have the weights first 
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #    --resume \
# #     --config configs/faster_rcnn_R101_cross_tless_incremental_22_from_21.yaml \
# #     SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #     SOLVER.BASE_LR 0.001 \
# #     SOLVER.MAX_ITER 50000 \
# #     SOLVER.WEIGHT_DECAY 0.001 \
# #     OUTPUT_DIR "./output-mymodel-classes-1-22-basedOn1-21-FINAL-MyModel_withCustomAugmentation" \
# #     SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #     SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #     SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #     INPUT.CROP.ENABLED False \
# #     INPUT.CUSTOM_AUGMENTATIONS True \
# #     MODEL.WEIGHTS output-mymodel-classes-1-20to1-20and21-FINAL-MyModel_withCustomAugmentation/model_best_wo_solver_states.pth

## train my custom model with original scheduler on pump images
srun python train_net_extended_class.py \
    --num-gpus 1 \
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

## For plotting training results
## srun python plot.py \
    # # --disable_student \
    # # --title AP50_for_class_21 \
    # # --path output-mymodel-classes-21only-FINAL-MyModel_withCustomAugmentation \
    # # --path_name Model_trained_on_class_21_only \
    # # --class_id  21 \
    # # --path output-mymodel-classes-1-30-FINAL-MyModel_withCustomAugmentation-1  \
    # # --path_name Model_trained_on_all_30_classes \
    # # --class_id 21 \
    # # --path output-mymodel-classes-1-20to1-20and21-FINAL-MyModel_withCustomAugmentation \
    # # --path_name Continual_learning_on_class_21 \
    # # --class_id  21 
    
## srun python plot.py \
    # # --disable_student \
    # # --path output-mymodel-classes-1-30-FINAL-MyModel_withCustomAugmentation-1  \
    # # --path_name Model_trained_on_all_30_classes \
    # # --class_id 0 \
    # # --path output-mymodel-classes-1-20to1-20and21-FINAL-MyModel_withCustomAugmentation \
    # # --path_name Continual_learning_on_class_21 \
    # # --class_id  0 
    
    # # \
    # # --path output-mymodel-classes-1-20to1-20and21-FINAL-MyModel_withCustomAugmentation \
    # # --path_name Continual_learning_for_21_classes \
    # # --class_id  0 


## srun python plot.py \
    # # --disable_student \
    # # --title Consistency_loss \
    # # --path output-mymodel-classes-1-30-FINAL-withCustomAugmentation \
    # # --path_name Lambda_=_0 \
    # # --class_id  0 \
    # # --path output-mymodel-classes-1-30-FINAL-MyModel_withCustomAugmentation-1 \
    # # --path_name Lambda_=_0.7 \
    # # --class_id 0 

####--------------------------------------------------------------------------
### RUN THIS ONE LAST TIME TO FIX THE AVERAGE PLOT!
## srun python plot.py \
    # # --disable_student \
    # # --title Continual_learning \
    # # --path output-mymodel-classes-1-20to1-20and21-FINAL-MyModel_withCustomAugmentation \
    # # --path_name Class_21 \
    # # --class_id  21 \
    # # --path output-mymodel-classes-1-20to1-20and22-FINAL-MyModel_withCustomAugmentation \
    # # --path_name Class_22 \
    # # --class_id  22 \
    # # --path output-mymodel-classes-1-20to1-20and23-FINAL-MyModel_withCustomAugmentation \
    # # --path_name Class_23 \
    # # --class_id  23 \
    # # --path output-mymodel-classes-1-20to1-20and24-FINAL-MyModel_withCustomAugmentation \
    # # --path_name Class_24 \
    # # --class_id  24 \
    # # --path output-mymodel-classes-1-20to1-20and25-FINAL-MyModel_withCustomAugmentation \
    # # --path_name Class_25 \
    # # --class_id  25 \
    # # --path output-mymodel-classes-1-20to1-20and26-FINAL-MyModel_withCustomAugmentation \
    # # --path_name Class_26 \
    # # --class_id  26\
    # # --path output-mymodel-classes-1-20to1-20and27-FINAL-MyModel_withCustomAugmentation \
    # # --path_name Class_27 \
    # # --class_id  27\
    # # --path output-mymodel-classes-1-20to1-20and28-FINAL-MyModel_withCustomAugmentation \
    # # --path_name Class_28 \
    # # --class_id  28\
    # # --path output-mymodel-classes-1-20to1-20and29-FINAL-MyModel_withCustomAugmentation \
    # # --path_name Class_29 \
    # # --class_id  29\
    # # --path output-mymodel-classes-1-20to1-20and30-FINAL-MyModel_withCustomAugmentation \
    # # --path_name Class_30 \
    # # --class_id  30


# # python plot.py \
# #     --title AP50 \
# #     --disable_student \
# #     --path output-mymodel-classes-1-30-FINAL-Original_withoutMyComponents \
# #     --path_name Original_AT \
# #     --class_id  0 \
# #     --path output-mymodel-classes-1-30-FINAL-Original_newScheduler \
# #     --path_name Original_AT_with_new_scheduler \
# #     --class_id  0 \
# #     --path output-mymodel-classes-1-30-FINAL-MyModel_withCustomAugmentation-origScheduler \
# #     --path_name Custom_AT_with_original_scheduler \
# #     --class_id  0 


## plot to show continual learning results for 21
# # python plot.py \
# #     --disable_student \
# #     --metric_name bbox_teacher/AP50 \
# #     --title Continual_learning \
# #     --path output-mymodel-classes-1-20to1-20and21-FINAL-MyModel_withCustomAugmentation \
# #     --path_name continual_learning_on_class_21 \
# #     --class_id  0 \
# #     --path output-mymodel-classes-1-30-FINAL-MyModel_withCustomAugmentation-1 \
# #     --path_name Trained_on_full_dataset \
# #     --class_id  0 

## for the pump
# # python plot.py \
# #     --disable_student \
# #     --metric_name bbox_teacher/AP50 \
# #     --title AP50_for_Pump_HM-75S \
# #     --path output-mymodel-pumps-FINAL-MyModel_withCustomAugmentation \
# #     --path_name AP50_for_Pump_HM-75S \
# #     --class_id  0 


# # python plot.py \
# #     --disable_student \
# #     --title Varying_lambda \
# #     --metric_name bbox/AP50 \
# #     --path output-mymodel-classes-1-20-constLoss_0.25 \
# #     --path_name L_const_0.25,_L_inst_0.05,_L_img_0.05,_BASE_LR_0.001\
# #     --class_id  0 \
# #     --path output-mymodel-classes-1-20-constLoss-v2 \
# #     --path_name L_const_0.07,_L_inst_0.07,_L_img_0.07,_BASE_LR_0.001\
# #     --class_id  0 \
# #     --path output-mymodel-classes-1-20-constLoss-v3 \
# #     --path_name L_const_0.1,_L_inst_0.05,_L_img_0.1,_BASE_LR_0.001\
# #     --class_id  0 \
# #     --path output-mymodel-classes-1-20-constLoss-v4 \
# #     --path_name L_const_0.1,_L_inst_0.1,_L_img_0.1,_BASE_LR_0.001\
# #     --class_id  0 \
# #     --path output-mymodel-classes-1-20-constLoss-v5 \
# #     --path_name L_const_0.1,_L_inst_0.1,_L_img_0.1,_BASE_LR_0.0012 \
# #     --class_id  0 \
# #     --path output-mymodel-classes-1-20-constLoss-v6.1 \
# #     --path_name L_const_0.1,_L_inst_0.125,_L_img_0.1,_BASE_LR_0.0008 \
# #     --class_id  0 


## srun python plot.py \
    # # --disable_student \
    # # --title Rich_distribution_vs_poor_distribution \
    # # --path output-mymodel-classes-1-30-FINAL-MyModel_withCustomAugmentation-origScheduler \
    # # --path_name AP50 \
    # # --class_id  0 
##################################################################
## These two are not up to date: 

## For evaluation:
##srun python train_net.py --eval-only --re --num-gpus 1 --config configs/faster_rcnn_R101_cross_tless.yaml MODEL.WEIGHTS output/model_final.pth

## For predictions + visual
##srun python predict.py --num-gpus 1 --config configs/faster_rcnn_R101_cross_tless.yaml MODEL.WEIGHTS output/model_final.pth
