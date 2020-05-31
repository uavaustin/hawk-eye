""" The augmentations used during training and inference for the various models. """

import albumentations


def clf_train_augs(height: int, width: int) -> albumentations.Compose:
    return albumentations.Compose(
        [
            albumentations.Resize(height=height, width=width),
            albumentations.Flip(),
            albumentations.Blur(blur_limit=3),
            albumentations.GaussNoise(),
            albumentations.HueSaturationValue(),
            albumentations.RandomBrightnessContrast(),
            albumentations.Normalize(),
        ]
    )

def clf_eval_augs(height: int, width: int) -> albumentations.Compose:
    return albumentations.Compose(
        [
            albumentations.Resize(height=height, width=width),
            albumentations.Normalize(),
        ]
    )
