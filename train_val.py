import time

import torch

from utils.loss_functions import get_dice_coeff
from utils.utils import AverageMeter, timeSince


def train_one_epoch(cfg, train_loader, model, criterion, optimizer, scaler, epoch, device, scheduler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    current_loss = AverageMeter()
    current_dice = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()

    # Iterate over dataloader
    for batch_idx, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # zero the gradients
        optimizer.zero_grad()

        images = images.to(device)
        labels = labels.to(device)

        batch_size = labels.size(0)

        if cfg.train_params.mixed_prec:
            # Runs the forward pass with autocasting
            with torch.cuda.amp.autocast():
                y_preds = model(images)
                loss = criterion(y_preds, labels)
                dice = get_dice_coeff(y_preds, labels)
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            # Updates the scale for next iteration.
            scaler.update()
        else:
            y_preds = model(images)
            loss = criterion(y_preds, labels)
            dice = get_dice_coeff(y_preds, labels)
            # Compute gradients and do step
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # record loss
        current_loss.update(loss.item(), batch_size)
        current_dice.update(dice, batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % cfg.train_params.print_freq == 0 or batch_idx == (len(train_loader) - 1):
            print(
                "Epoch: [{Epoch:d}][{Iter:d}/{Len:d}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                    Epoch=epoch + 1,
                    Iter=batch_idx,
                    Len=len(train_loader),
                    data_time=data_time,
                    loss=current_loss,
                    remain=timeSince(start, float(batch_idx + 1) / len(train_loader)),
                )
            )
    return current_loss.avg, current_dice.avg


def valid_one_epoch(cfg, valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    current_loss = AverageMeter()
    current_dice = AverageMeter()
    # switch to evaluation mode
    model.eval()
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds, labels)
        dice = get_dice_coeff(y_preds, labels)
        current_loss.update(loss.item(), batch_size)
        current_dice.update(dice, batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % cfg.train_params.print_freq == 0 or step == (len(valid_loader) - 1):
            print(
                "EVAL: [{Step:d}/{Len:d}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                    Step=step,
                    Len=len(valid_loader),
                    data_time=data_time,
                    loss=current_loss,
                    remain=timeSince(start, float(step + 1) / len(valid_loader)),
                )
            )

    return current_loss.avg, current_dice.avg
