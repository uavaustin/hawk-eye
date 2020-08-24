""" The augmentations used during training and inference for the various models. These
are not meant to be an all-encompassing augmentation regime for any of the models.
Feel free to experiment with any of the available augmentations:
https://albumentations.readthedocs.io/en/latest/index.html """

import albumentations as albu


def clf_train_augs(height: int, width: int) -> albu.Compose:
    return albu.Compose(
        [
            albu.Resize(height=height, width=width),
            albu.OneOf(
                [
                    alb.IAAAffine(shear=6, rotate=5, always_apply=True),
                    albu.ShiftScaleRotate(
                        shift_limit=0.025, scale_limit=0.1, rotate_limit=10
                    ),
                ]
            ),
            albu.ShiftScaleRotate(shift_limit=0.025, scale_limit=0.1, rotate_limit=10),
            albu.Flip(),
            albu.RandomRotate90(),
            albu.OneOf(
                [
                    albu.HueSaturationValue(p=1.0),
                    albu.IAAAdditiveGaussianNoise(p=1.0),
                    albu.IAASharpen(p=1.0),
                    albu.RandomBrightnessContrast(
                        brightness_limit=0.1, contrast_limit=0.1, p=1.0
                    ),
                    albu.RandomGamma(p=1.0),
                ],
                p=1.0,
            ),
            albu.OneOf(
                [
                    albu.Blur(blur_limit=3, p=1.0),
                    albu.MedianBlur(blur_limit=3, p=1.0),
                    albu.MotionBlur(blur_limit=3, p=1.0),
                ],
                p=1.0,
            ),
            albu.Normalize(),
        ]
    )


def clf_eval_augs(height: int, width: int) -> albu.Compose:
    return albu.Compose([albu.Resize(height=height, width=width), albu.Normalize()])


# TODO(alex): Add some more augumentations here.
def det_train_augs(height: int, width: int) -> albu.Compose:
    return albu.Compose(
        [
            albu.Resize(height=height, width=width),
            albu.ShiftScaleRotate(shift_limit=0.025, scale_limit=0.1, rotate_limit=10),
            albu.Flip(),
            albu.RandomRotate90(),
            albu.OneOf(
                [
                    albu.HueSaturationValue(p=1.0),
                    albu.IAAAdditiveGaussianNoise(p=1.0),
                    albu.IAASharpen(p=1.0),
                    albu.RandomBrightnessContrast(
                        brightness_limit=0.1, contrast_limit=0.1, p=1.0
                    ),
                    albu.RandomGamma(p=1.0),
                ],
                p=1.0,
            ),
            albu.OneOf(
                [albu.Blur(blur_limit=3, p=1.0), albu.MotionBlur(blur_limit=3, p=1.0)],
                p=1.0,
            ),
            albu.Normalize(),
        ]
    )


def det_val_augs(height: int, width: int) -> albu.Compose:
    return albu.Compose([albu.Resize(height=height, width=width), albu.Normalize()])


def feature_extraction_aug(height: int, width: int) -> albu.Compose:
    return albu.Compose(
        [
            albu.Resize(height=height, width=width),
            albu.Rotate(5),
            albu.RandomBrightnessContrast(0.05, 0.05),
            albu.GaussianBlur(blur_limit=4),
            albu.Normalize(),
        ]
    )
