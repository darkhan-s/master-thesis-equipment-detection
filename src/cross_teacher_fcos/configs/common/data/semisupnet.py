from omegaconf import OmegaConf

semisupnet = OmegaConf.create()

# Output dimension of the MLP projector after `res5` block
semisupnet.mlp_dim = 128
semisupnet.trainer = "crossteacher"
semisupnet.bbox_threshold = 0.7
semisupnet.pseudo_bbox_sample = "thresholding"
semisupnet.teacher_update_iter = 1
semisupnet.burn_up_step = 12000
semisupnet.unsup_loss_weight = 0.5
semisupnet.loss_weight_type = "standard"
semisupnet.dis_type = "res4"
semisupnet.dis_loss_weight = 0.1


