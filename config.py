import numpy as np


class CFG:
    mean = np.array([0.65459856, 0.48386562, 0.69428385])
    std = np.array([0.15167958, 0.23584107, 0.13146145])
    data = 256
    debug = False
    apex = False
    print_freq = 100
    num_workers = 4
    img_size = 224  # appropriate input size for encoder
    scheduler = (
        "CosineAnnealingWarmRestarts"  # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    )
    epoch = 5  # Change epochs
    criterion = "Lovasz"  # 'DiceBCELoss'  ['DiceLoss', 'Hausdorff', 'Lovasz']
    base_model = "FPN"  # ['Unet']
    encoder = "se_resnet50"  # ['efficientnet-b5'] or other encoders from smp
    lr = 1e-4
    min_lr = 1e-6
    batch_size = 4
    weight_decay = 1e-6
    gradient_accumulation_steps = 1
    seed = 2021
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]
    train = True
    inference = False
    optimizer = "Adam"
    T_0 = 10
    # N=5
    # M=9
    T_max = 10
    # factor=0.2
    # patience=4
    # eps=1e-6
    smoothing = 1
