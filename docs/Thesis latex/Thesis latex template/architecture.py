## image-level domain adaptation
(RCNN_Image_DA): 
(
    (conv1): Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (classifier): Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (leaky_relu): LeakyReLU(negative_slope=0.2, inplace=True)
)
## instance-level domain adaptation
(RCNN_Instance_DA): 
(
    (fc_1): Linear(in_features=1024, out_features=256, bias=True)
    (leaky_relu_1): LeakyReLU(negative_slope=0.2, inplace=True)
    (dropout_1): Dropout(p=0.5, inplace=False)
    (fc_2): Linear(in_features=256, out_features=256, bias=True)
    (leaky_relu_2): LeakyReLU(negative_slope=0.2, inplace=True)
    (dropout_2): Dropout(p=0.5, inplace=False)
    (classifier): Linear(in_features=256, out_features=1, bias=True)
    (leaky_relu_3): LeakyReLU(negative_slope=0.2, inplace=True)
)
## consistency loss term between the two alignments
(consistency_loss): MSELoss()

## module for continual learning
(FastRCNNExtendedOutputLayers): 
(
      ## for the original 20 classes and the background
      (cls_score): Linear(in_features=1024, out_features=21, bias=True) 
      (bbox_pred): Linear(in_features=1024, out_features=80, bias=True) 
      (cls_score_extra_classes): Linear(in_features=1024, out_features=2, bias=True)
       ## for one added class and the background
      (bbox_pred_extra_classes): Linear(in_features=1024, out_features=4, bias=True)
)