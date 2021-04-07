import time


def train_one_epoch(epoch, model, device, optimizer, scheduler, trainloader, criterion):
    model.train()
    t = time.time()
    total_loss = 0
    for step, (images, targets) in enumerate(trainloader):
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        if (step + 1) % 4 == 0 or (step + 1) == len(trainloader):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        loss = loss.detach().item()
        total_loss += loss
        if (step + 1) % 10 == 0 or (step + 1) == len(trainloader):
            print(
                f"epoch {epoch} train step {step + 1}/{len(trainloader)}, "
                + f"loss: {total_loss / len(trainloader):.4f}, "
                + f"time: {(time.time() - t):.4f}",
                end="\r" if (step + 1) != len(trainloader) else "\n",
            )


def valid_one_epoch(epoch, model, device, optimizer, scheduler, validloader, criterion):
    model.eval()
    t = time.time()
    total_loss = 0
    for step, (images, targets) in enumerate(validloader):
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss = loss.detach().item()
        total_loss += loss
        if (step + 1) % 4 == 0 or (step + 1) == len(validloader):
            scheduler.step(total_loss / len(validloader))
        if (step + 1) % 10 == 0 or (step + 1) == len(validloader):
            print(
                f"epoch {epoch} val step {step + 1}/{len(validloader)}, "
                + f"loss: {total_loss / len(validloader):.4f}, "
                + f"time: {(time.time() - t):.4f}",
                end="\r" if (step + 1) != len(validloader) else "\n",
            )
