# Experiments

## Baseline
* tile size = 256
* batch size = 64
* Adam: lr = 0.0001; wd = 1e-4; eps = 1e-4
* Scheduler: None
* 5 fold validation
* Criterion: BCEWithLogitsLoss
* Metric: Dice

## Trying Unet with different encoders
### Resnet34
Dice scores for folds:

Fold 1, 2, 3, 4, 5 = [0.9323, 0.9457, 0.9428, 0.9505, 0.9508]

Avg score = 0.9444

Public score = 0.85

### Se_resnext50_32x4d
Dice scores for folds:

Fold 1, 2, 3, 4, 5 = [0.9339, 0.9507, 0.9503, 0.9579, 0.9557]

Avg score = 0.9497

Public score = 0.866

### timm-efficientnet-b4
Dice scores for folds:

Fold 1, 2, 3, 4, 5 = [0.9331, 0.9476, 0.9473, 0.9528, 0.9512]

Avg score = 0.9464
Public score = 0.854

## Trying larger tile sizes
### 512x512
Dice scores for folds:

Fold 1, 2, 3, 4, 5 = [0.93512714, 0.94898659, 0.94976795 ,0.95760339, 0.95456284]

Avg score = 0.9492
Public score = 0.834

### 1024x1024
Dice scores for folds:

Fold 1, 2, 3, 4, 5 = [0.86785412, 0.88530695, 0.91538972, 0.82604915, 0.93004316]

Avg score = 0.8849
Public score = 0.834

## Trying different loss functions
### Dice loss
Dice scores for folds:

Fold 1, 2, 3, 4, 5 = [0.93481171, 0.93642789, 0.94222426, 0.94669557, 0.94715351]

Avg score = 0.9415
Public score = 0.835

### Jaccard loss
Dice scores for folds:

Fold 1, 2, 3, 4, 5 = [0.93685228, 0.93933076, 0.93893105, 0.9465633, 0.94640851]

Avg score = 0.9416
Public score = 0.829

### Dice + BCE loss (0.5 and 1 coefficients)
