# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torchvision.transforms as transforms
from adapteacher.data.transforms.augmentation_impl import (
    GaussianBlur,
)


def build_strong_augmentation(cfg, is_train):
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    @darkhan-s equalizing and AugMix added
    Returns:
        list[Augmentation]
    """

    logger = logging.getLogger(__name__)
    augmentation = []
    if is_train:
        # This is simialr to SimCLR https://arxiv.org/abs/2002.05709
        augmentation.append(
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.7)
        )
        augmentation.append(transforms.RandomGrayscale(p=0.2))
        augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))
        
        
        # @darkhan-s custom augmentations 
        if cfg.INPUT.CUSTOM_AUGMENTATIONS:
            augmentation.append(transforms.RandomEqualize(p=0.3))
            augmentation.append(transforms.RandomApply(transforms=[transforms.RandomAdjustSharpness(sharpness_factor=100)], p=0.3))


        # @darkhan-s add random rotation to weak augmentation (hard coded between -15,+15 degrees for now)
        # have to rotate the annotations too + takes more memory if in use so disabled for now
        # if cfg.INPUT.ROTATION_ENABLED:
        #     augmentation.append(transforms.RandomRotation(angle=[-10, 10]))

        ## @darkhan-s adding random affine transformation (hard coded params for now)
        ## seems to worsen the results because have to adjust labels
        # if cfg.INPUT.AFFINE_ENABLED:
        #     augmentation.append(transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)))

        randcrop_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomErasing(
                    p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"
                ),
                transforms.RandomErasing(
                    p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"
                ),
                transforms.RandomErasing(
                    p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"
                ),
                transforms.ToPILImage(),
            ]
        )
        augmentation.append(randcrop_transform)

        logger.info("Augmentations used in training: " + str(augmentation))
    return transforms.Compose(augmentation)
