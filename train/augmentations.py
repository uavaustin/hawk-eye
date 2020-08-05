""" The augmentations used during training and inference for the various models. These
are not meant to be an all-encompassing augmentation regime for any of the models.
Feel free to experiment with any of the available augmentations:
https://albumentations.readthedocs.io/en/latest/index.html """

import albumentations


def clf_train_augs(height: int, width: int) -> albumentations.Compose:
    return albumentations.Compose(
        [
            albumentations.Resize(height=height, width=width),
            albumentations.Flip(),
            albumentations.OneOf(
                [
                    albumentations.HueSaturationValue(),
                    albumentations.RandomBrightnessContrast(),
                    albumentations.Blur(blur_limit=3),
                    albumentations.GaussNoise(),
                ]
            ),
            albumentations.Normalize(),
        ]
    )


def clf_eval_augs(height: int, width: int) -> albumentations.Compose:
    return albumentations.Compose(
        [albumentations.Resize(height=height, width=width), albumentations.Normalize()]
    )


# TODO(alex): Add some more augumentations here.
def det_train_augs(height: int, width: int) -> albumentations.Compose:
    return albumentations.Compose(
        [
            albumentations.Resize(height=height, width=width),
            albumentations.OneOf(
                [
                    albumentations.RandomBrightnessContrast(),
                    albumentations.GaussNoise(),
                    albumentations.HueSaturationValue(),
                ],
                p=1.0,
            ),
            albumentations.Flip(),
            albumentations.RandomRotate90(),
            albumentations.Normalize(),
        ]
    )


def det_eval_augs(height: int, width: int) -> albumentations.Compose:
    return albumentations.Compose(
        [albumentations.Resize(height=height, width=width), albumentations.Normalize()]
    )


def feature_extraction_augmentations(height: int, width: int) -> albumentations.Compose:
    return albumentations.Compose(
        [
            albumentations.Resize(height=height, width=width),
            albumentations.Rotate(5),
            albumentations.RandomBrightnessContrast(0.05, 0.05),
            albumentations.GaussianBlur(blur_limit=4),
            albumentations.Normalize(),
        ]
    )
