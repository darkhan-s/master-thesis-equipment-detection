#!/bin/bash
#SBATCH --job-name=detectron2TrainCrossTeacher
#SBATCH --account=project_2005695
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
##SBATCH --mail-type=ALL
##SBATCH --output=slurm-detectron2-adapt-output_%j.txt
##SBATCH --error=slurm-detectron2-adapt-errors_%j.txt
#SBATCH --mem 8G 

### For training: 
##SBATCH --partition=gpusmall
##SBATCH --time=12:30:00

### For testing: 
#SBATCH --partition=gputest
#SBATCH --time=00:15:00


ml purge
##ml pytorch/1.10

source /projappl/project_2005695/miniconda3/etc/profile.d/conda.sh
conda activate base
##conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 opencv -c pytorch -c conda-forge
export CUDA_LAUNCH_BLOCKING=1
##export TORCH_DISTRIBUTED_DEBUG=DETAIL
##python -m pip install detectron2 --user -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html


### For training: 
##srun python /scratch/project_2005695/master-thesis-equipment-detection/src/cross_teacher_fcos/fcos_train_net.py --config-file /scratch/project_2005695/master-thesis-equipment-detection/src/cross_teacher_fcos/configs/fcos_R_50_FPN_1x.py  

rm -rf ./output-mymodel-quickTest


############## FINAL FULL TRAINING ############### 
##srun python train_net_extended_class.py --num-gpus 1 --config configs/faster_rcnn_R101_cross_tless.yaml OUTPUT_DIR "./output-mymodel-quickTest" SOLVER.MAX_ITER 3000 DATALOADER.DEBUG_LIMIT_INPUT 300 INPUT.CROP.ENABLED True SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.0 SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.0 SEMISUPNET.DIS_LOSS_WEIGHT 0.05
# # srun python train_net_extended_class.py \
# #     --num-gpus 1 \
# #   --config configs/faster_rcnn_R101_cross_tless.yaml \
# #   OUTPUT_DIR "./output-mymodel-quickTest" \
# #   SOLVER.MAX_ITER 3000 \
# #   DATALOADER.DEBUG_LIMIT_INPUT 300 \
# #   INPUT.CROP.ENABLED True \
# #   SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.0 \
# #   SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.0 \
# #   SEMISUPNET.DIS_LOSS_WEIGHT 0.05


############## FINAL RESUME TRAINING ############### 
##srun python train_net_extended_class.py --num-gpus 1 --resume --config configs/faster_rcnn_R101_cross_tless_incremental.yaml OUTPUT_DIR "./output-mymodel-quickTest"  MODEL.ROI_HEADS.NUM_CLASSES 20 MODEL.WEIGHTS output-mymodel-classes-0-20/model_best_wo_solver_states.pth SOLVER.MAX_ITER 3000 DATALOADER.DEBUG_LIMIT_INPUT 300 
##srun python train_net_extended_class.py --num-gpus 1 --resume --config configs/faster_rcnn_R101_cross_tless_incremental.yaml OUTPUT_DIR  "./output-mymodel-quickTest"  MODEL.ROI_HEADS.NUM_CLASSES 20 MODEL.WEIGHTS output-mymodel-classes-1-20-constLoss-v6.1/model_best_wo_solver_states.pth SOLVER.MAX_ITER 17000 DATALOADER.DEBUG_LIMIT_INPUT 300 

# # srun python train_net_extended_class.py \
# #   --num-gpus 1 \
# #   --config configs/faster_rcnn_R101_cross_tless_one_class.yaml \
# #   SOLVER.LR_SCHEDULER_NAME "LRMultiplier" \
# #   SOLVER.BASE_LR 0.001 \
# #   SOLVER.MAX_ITER 50000 \
# #   SOLVER.WEIGHT_DECAY 0.001 \
# #   OUTPUT_DIR "./output-mymodel-quickTest" \
# #   SEMISUPNET.CONSISTENCY_LOSS_WEIGHT 0.07 \
# #   SEMISUPNET.DIS_INST_LOSS_WEIGHT 0.07 \
# #   SEMISUPNET.DIS_LOSS_WEIGHT 0.07 \
# #   SOLVER.BEST_CHECKPOINTER.PATIENCE 10 \
# #   INPUT.CROP.ENABLED False \
# #   INPUT.CUSTOM_AUGMENTATIONS True 
# #   SOLVER.MAX_ITER 3000 \
# #   DATALOADER.DEBUG_LIMIT_INPUT 300 


##srun python train_net_incremental.py --num-gpus 1 --config configs/faster_rcnn_R101_cross_tless.yaml OUTPUT_DIR "./output-mymodel-quickTest" INPUT.CROP.ENABLED False INPUT.ROTATION_ENABLED False INPUT.AFFINE_ENABLED False SOLVER.BASE_LR 0.001 DATALOADER.LIMIT_CLASSES 0,1 MODEL.ROI_HEADS.NUM_CLASSES 1 SOLVER.MAX_ITER 10000 DATALOADER.DEBUG_LIMIT_INPUT 300

##srun python train_net_incremental.py --num-gpus 1 --resume --config configs/faster_rcnn_R101_cross_tless.yaml OUTPUT_DIR "./output-mymodel-quickTest" INPUT.CROP.ENABLED False INPUT.ROTATION_ENABLED False INPUT.AFFINE_ENABLED False SOLVER.BASE_LR 0.0004 DATALOADER.LIMIT_CLASSES 20,30 MODEL.ROI_HEADS.NUM_CLASSES 30 SOLVER.MAX_ITER 20000 MODEL.WEIGHTS extended_models/model_extended_best.pth DATALOADER.DEBUG_LIMIT_INPUT 300

##srun python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:53372 --config configs/faster_rcnn_R101_cross_tless.yaml OUTPUT_DIR "./output-mymodel-quickTest" INPUT.CROP.ENABLED False INPUT.ROTATION_ENABLED False INPUT.AFFINE_ENABLED False SOLVER.BASE_LR 0.001 DATALOADER.DEBUG_LIMIT_INPUT 300 DATALOADER.LIMIT_CLASSES 0,20 MODEL.ROI_HEADS.NUM_CLASSES 20 SOLVER.MAX_ITER 20000
##srun python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:53372 --config configs/faster_rcnn_R101_cross_tless.yaml OUTPUT_DIR "./output-mymodel-quickTest" INPUT.CROP.ENABLED False INPUT.ROTATION_ENABLED False INPUT.AFFINE_ENABLED False SOLVER.BASE_LR 0.001 DATALOADER.DEBUG_LIMIT_INPUT 300 DATALOADER.LIMIT_CLASSES 20,30 MODEL.ROI_HEADS.NUM_CLASSES 10 SOLVER.MAX_ITER 20000
##srun python train_net.py --resume --num-gpus 2 --config configs/faster_rcnn_R101_cross_tless.yaml  MODEL.WEIGHTS output/model_best.pth




## for training with pumps
srun python train_net_extended_class.py --num-gpus 1 --config configs/faster_rcnn_R101_cross_pump.yaml OUTPUT_DIR "./output-mymodel-quickTest"  DATALOADER.DEBUG_LIMIT_INPUT 300 


## For plotting training results
##srun python plot.py --path output

### For evaluation:
##srun python train_net.py --eval-only --num-gpus 1 --config configs/faster_rcnn_R101_cross_tless.yaml MODEL.WEIGHTS output-mymodel-classes-20-30/model_final.pth DATALOADER.LIMIT_CLASSES 0,30 MODEL.ROI_HEADS.NUM_CLASSES 30
# srun python train_net.py --eval-only --num-gpus 1 --config configs/faster_rcnn_R101_cross_tless.yaml MODEL.WEIGHTS output-mymodel-classes-0-20/model_best.pth DATALOADER.LIMIT_CLASSES 0,20 MODEL.ROI_HEADS.NUM_CLASSES 20
# srun python train_net.py --eval-only --num-gpus 1 --config configs/faster_rcnn_R101_cross_tless.yaml MODEL.WEIGHTS output-mymodel-classes-0-20/model_best.pth DATALOADER.LIMIT_CLASSES 0,30 MODEL.ROI_HEADS.NUM_CLASSES 30
# srun python train_net.py --eval-only --num-gpus 1 --config configs/faster_rcnn_R101_cross_tless.yaml MODEL.WEIGHTS output-mymodel-classes-20-30/model_best.pth DATALOADER.LIMIT_CLASSES 0,30 MODEL.ROI_HEADS.NUM_CLASSES 30

## For predictions + visual
##srun python predict.py --num-gpus 1 --config configs/faster_rcnn_R101_cross_tless.yaml MODEL.WEIGHTS output/model_final.pth


## For interactive commands, example: 
## sinteractive --account project_2005695 --time 00:15:00 --cores 6 python train_net_extended_class.py --resume --dist-url tcp://127.0.0.1:53385 --config configs/faster_rcnn_R101_cross_tless_incremental.yaml OUTPUT_DIR "./output-mymodel-quickTest"  DATALOADER.LIMIT_CLASSES 1,2 MODEL.ROI_HEADS.NUM_CLASSES 1 MODEL.DEVICE cpu MODEL.WEIGHTS output-mymodel-classes-1/model_best_wo_solver_states.pth DATALOADER.DEBUG_LIMIT_INPUT 300 SOLVER.BASE_LR 0.1

## interactive, 1 class, cpu test
## sinteractive --account project_2005695 --time 1:00:00 --cores 16 python train_net.py --num-gpus 1 --resume --config configs/faster_rcnn_R101_cross_tless.yaml OUTPUT_DIR "./output-mymodel-quickTest"  DATALOADER.LIMIT_CLASSES 0,1 MODEL.ROI_HEADS.NUM_CLASSES 1 MODEL.WEIGHTS "./output-mymodel-classes-1/model_best_wo_solver_states.pth" SOLVER.MAX_ITER 15 DATALOADER.DEBUG_LIMIT_INPUT 100 MODEL.DEVICE cpu

## interactive, extended classes, cpu test
## sinteractive --account project_2005695 --time 1:00:00 --cores 16 python train_net_extended_class.py --num-gpus 1 --resume --config configs/faster_rcnn_R101_cross_tless_incremental.yaml OUTPUT_DIR "./output-mymodel-quickTest"  DATALOADER.LIMIT_CLASSES 1,3 MODEL.ROI_HEADS.NUM_CLASSES 1 MODEL.WEIGHTS output-mymodel-classes-1/model_best_wo_solver_states.pth SOLVER.MAX_ITER 15 DATALOADER.DEBUG_LIMIT_INPUT 300 MODEL.DEVICE cpu