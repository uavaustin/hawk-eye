""" The augmentations used during training and inference for the various models. These
are not meant to be an all-encompassing augmentation regime for any of the models.
Feel free to experiment with any of the available augmentations:
https://albumentations.readthedocs.io/en/latest/index.html """

import albumentations as albu


def clf_train_augs(height: int, width: int) -> albu.Compose:
    return albu.Compose(
        [
            albu.Resize(height=height, width=width),
            albu.Flip(),
            albu.RandomRotate90(),
            albu.OneOf(
                [
                    albu.HueSaturationValue(),
                    albu.RandomBrightnessContrast(),
                    albu.Blur(blur_limit=2),
                    albu.GaussNoise(),
                    albu.RandomGamma(),
                ]
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
            albu.RandomGamma(),
            albu.Flip(),
            albu.RandomRotate90(),
            albu.Normalize(),
        ],
        albu.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
    )


def det_eval_augs(height: int, width: int) -> albu.Compose:
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
