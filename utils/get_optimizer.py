import torch


def get_optimizer(cfg, model):
    parameters = model.parameters()

    if cfg.optimizer.type == "SGD":
        optimizer = torch.optim.SGD(
            parameters,
            lr=cfg.optimizer.lr,
            momentum=cfg.optimizer.momentum,
            weight_decay=cfg.optimizer.weight_decay,
            nesterov=cfg.optimizer.nesterov,
        )
    elif cfg.optimizer.type == "Adam":
        optimizer = torch.optim.Adam(
            parameters, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay, eps=cfg.optimizer.eps
        )
    else:
        assert False and "WTF optimizer?"
        raise ValueError
    return optimizer
