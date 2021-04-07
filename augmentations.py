import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from config import CFG

base_transform = A.Compose(
    [
        A.Resize(CFG.img_size, CFG.img_size, p=1.0),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=0.9, border_mode=cv2.BORDER_REFLECT
        ),
        A.OneOf(
            [
                A.OpticalDistortion(p=0.4),
                A.GridDistortion(p=0.1),
                A.IAAPiecewiseAffine(p=0.4),
            ],
            p=0.3,
        ),
        A.OneOf(
            [
                A.HueSaturationValue(10, 15, 10),
                A.CLAHE(clip_limit=3),
                A.RandomBrightnessContrast(),
            ],
            p=0.5,
        ),
        ToTensorV2(),
    ],
    p=1.0,
)

strong_transform = A.Compose(
    [
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        A.OneOf([A.RandomGamma(), A.GaussNoise()], p=0.5),
        A.OneOf(
            [
                A.OpticalDistortion(p=0.4),
                A.GridDistortion(p=0.2),
                A.IAAPiecewiseAffine(p=0.4),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.HueSaturationValue(10, 15, 10),
                A.CLAHE(clip_limit=4),
                A.RandomBrightnessContrast(),
            ],
            p=0.5,
        ),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        A.Resize(CFG.img_size, CFG.img_size, p=1.0),
        ToTensorV2(),
    ]
)

weak_transform = A.Compose(
    [
        A.Resize(CFG.img_size, CFG.img_size, p=1.0),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9, border_mode=cv2.BORDER_REFLECT
        ),
        ToTensorV2(),
    ],
    p=1.0,
)

val_transform = A.Compose([A.Resize(CFG.img_size, CFG.img_size, p=1.0), ToTensorV2()], p=1.0)
