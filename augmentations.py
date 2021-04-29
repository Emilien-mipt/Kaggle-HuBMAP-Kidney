import albumentations as A
import cv2


def get_aug(p=1.0):
    return A.Compose(
        [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9, border_mode=cv2.BORDER_REFLECT
            ),
            A.OneOf(
                [
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.1),
                    A.IAAPiecewiseAffine(p=0.3),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.HueSaturationValue(10, 15, 10),
                    A.CLAHE(clip_limit=2),
                    A.RandomBrightnessContrast(),
                ],
                p=0.3,
            ),
        ],
        p=p,
    )
