import time

import torch

from tensorboard_logger import log_value

from util import AverageMeter


def _accuracy(output, target, top_k=(1,)):
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


def _adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def lr_multiplier_functor(batches_per_epoch, base_lr=1.0, warmup_iters=0, milestones=(30, 60, 90), gamma=0.1):
    def calculate_lr(epoch, current_epoch_iter):
        lr = base_lr
        total_iter = current_epoch_iter + epoch * batches_per_epoch
        if total_iter < warmup_iters:
            return lr * total_iter / warmup_iters
        for milestone in milestones:
            if epoch >= milestone:
                lr *= gamma
        return lr

    return calculate_lr


def train(train_loader, model, criterion, optimizer, epoch, num_epochs,
          is_koala=False, print_freq=50, calculate_lr=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if is_koala:
            optimizer.predict()

        # compute output and loss
        output = model(input)
        loss = criterion(output, target)
        loss_mean = loss.mean()

        # compute gradient and perform update step
        if calculate_lr is not None:
            _adjust_learning_rate(optimizer, calculate_lr(epoch, i))
        optimizer.zero_grad()
        loss_mean.backward()
        if is_koala:
            loss_var = torch.mean(torch.pow(loss, 2))
            optimizer.update(loss_mean, loss_var)
        else:
            optimizer.step()

        # measure accuracy and record loss
        err1, err5 = _accuracy(output.data, target, top_k=(1, 5))
        losses.update(loss_mean.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(f'Epoch: [{epoch}/{num_epochs}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.get_last_val():.3f} ({batch_time.get_avg():.3f})\t'
                  f'Data {data_time.get_last_val():.3f} ({data_time.get_avg():.3f})\t'
                  f'Loss {losses.get_last_val():.4f} ({losses.get_avg():.4f})\t'
                  f'Top 1-err {top1.get_last_val():.4f} ({top1.get_avg():.4f})\t'
                  f'Top 5-err {top5.get_last_val():.4f} ({top5.get_avg():.4f})')

            # log to TensorBoard
            log_value('train_loss', losses.get_last_val(), epoch * len(train_loader) + i)
            log_value('train_error', top1.get_last_val(), epoch * len(train_loader) + i)


def validate(val_loader, model, criterion, epoch, num_epochs, print_freq=50):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            output = model(input)
            loss = torch.mean(criterion(output, target))

        # measure accuracy and record loss
        err1, err5 = _accuracy(output.data, target, top_k=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(f'Test (on val set): [{epoch}/{num_epochs}][{i}/{len(val_loader)}]\t'
                  f'Time {batch_time.get_last_val():.3f} ({batch_time.get_avg():.3f})\t'
                  f'Loss {losses.get_last_val():.4f} ({losses.get_avg():.4f})\t'
                  f'Top 1-err {top1.get_last_val():.4f} ({top1.get_avg():.4f})\t'
                  f'Top 5-err {top5.get_last_val():.4f} ({top5.get_avg():.4f})')

    print(f'* Epoch: [{epoch}/{num_epochs}]\t'
          f'Top 1-err {top1.get_avg():.3f}\t'
          f'Top 5-err {top5.get_avg():.3f}\t'
          f'Test Loss {losses.get_avg():.3f}')

    # log to TensorBoard
    log_value('val_loss', losses.get_avg(), epoch)
    log_value('val_top1', top1.get_avg(), epoch)
    log_value('val_top5', top5.get_avg(), epoch)

    return top1.get_avg(), top5.get_avg()
