# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from typing import Dict, List, Optional, Tuple, Union
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
)
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from adapteacher.modeling.roi_heads.fast_rcnn import FastRCNNFocaltLossOutputLayers, FastRCNNExtendedOutputLayers

import numpy as np
from detectron2.modeling.poolers import ROIPooler

from detectron2.modeling.sampling import subsample_labels

@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsPseudoLab(StandardROIHeads):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )
        
        if cfg.MODEL.ROI_HEADS.LOSS == "CrossEntropy":
            
            ## @darkhan-s for continual learning (expanding network)
            classesAdded = []
            for element in cfg.MODEL.ROI_HEADS.NEW_CLASSES:
                if element not in cfg.MODEL.ROI_HEADS.OLD_CLASSES:
                    classesAdded.append(element)
            
            if len(classesAdded) > 0:
                print(f"New classes are requested to train: {classesAdded}. Adding extra classifier layer..")
                box_predictor = FastRCNNExtendedOutputLayers(cfg, box_head.output_shape)
            else: 
                # regular prediction output
                box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)


        elif cfg.MODEL.ROI_HEADS.LOSS == "FocalLoss":
            box_predictor = FastRCNNFocaltLossOutputLayers(cfg, box_head.output_shape)

            for element in cfg.MODEL.ROI_HEADS.NEW_CLASSES:
                if element not in cfg.MODEL.ROI_HEADS.OLD_CLASSES:
                    raise NotImplementedError("Extra classifier not implemented for this head loss.")
        else:
            raise ValueError("Unknown ROI head loss.")


        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        compute_loss=True,
        branch="",
        compute_val_loss=False,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        del images
        if self.training and compute_loss:  # apply if training loss
            assert targets
            # 1000 --> 512
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )
        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )  # do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt
        del targets

        if (self.training and compute_loss) or compute_val_loss:
            losses, _, box_features = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )
            return proposals, losses, box_features
        else:
            pred_instances, predictions, box_features = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )

            return pred_instances, predictions, box_features

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        compute_loss: bool = True,
        compute_val_loss: bool = False,
        branch: str = "",
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        #@ darkhan-s dont delete box features because they are needed for instance level alignment
        #del box_features
        if (self.training and compute_loss) or compute_val_loss:  # apply if training loss or val loss
            losses = self.box_predictor.losses(predictions, proposals)

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions, box_features
        else:

            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, predictions, box_features

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], branch: str = ""
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes
            
            # check if  extra classes are added
            try:
                num_bg_samples.append((gt_classes == (self.num_classes + self.box_predictor.extra_num_classes)).sum().item())
            except AttributeError:
                num_bg_samples.append((gt_classes == self.num_classes).sum().item())

            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        storage = get_event_storage()

        if branch != "val_loss":
            storage.put_scalar(
                "roi_head/num_target_fg_samples_" + branch, np.mean(num_fg_samples)
            )
            storage.put_scalar(
                "roi_head/num_target_bg_samples_" + branch, np.mean(num_bg_samples)
            )

        return proposals_with_gt

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            try:
                gt_classes[matched_labels == 0] = (self.num_classes + self.box_predictor.extra_num_classes)
            except AttributeError:
                gt_classes[matched_labels == 0] = self.num_classes

            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            try:
                gt_classes = torch.zeros_like(matched_idxs) + (self.num_classes + self.box_predictor.extra_num_classes)
            except AttributeError:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        try:
            sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
                gt_classes, self.batch_size_per_image, self.positive_fraction, (self.num_classes + self.box_predictor.extra_num_classes)
            )
        except AttributeError: 
            sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
                gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
            )            

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]