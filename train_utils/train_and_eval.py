import torch
from torch import nn
import train_utils.distributed_utils as utils
import torch.nn.functional as F


def criterion(inputs, target):
    #losses = {}
    # for name, x in inputs.items():
    #     # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
    #     losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)
    losses = nn.functional.cross_entropy(inputs, target)
    # if len(losses) == 1:
    #     return losses['out']

    # return losses['out'] + 0.5 * losses['aux']
    return losses

def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            #output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()
        a = confmat
        b = a
    d = 1
    e = 2

    return confmat


def  train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        output = model(image)
        # a = output.softmax(dim=1)
        # output = a.argmax(dim=1)
        optimizer.zero_grad()
        loss = F.cross_entropy(output, target,ignore_index=-1)
        # loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
