import os
import wandb
import torch
import math
import numpy as np
import datetime

def make_dir(dir):
    if os.path.exists(dir) == False:
        os.mkdir(dir)

def set_dir():
    make_dir('./result')
    dataset_list = ['cifar10', 'cifar100', 'BAPPS', 'stl10']
    distortions = ['supcon', 'rand', 'v1', 'v2', 'v3']
    model_list = ['resnet18', 'resnet34', 'resnet50', 'vgg16', 'vgg19']
    for t in distortions:
        make_dir(f'./result/{t}')
        for d in dataset_list:
            make_dir(f'./result/{t}/{d}')
            for m in model_list:
                make_dir(f'./result/{t}/{d}/{m}')

def set_optimizer(opt, model):
    optimizer = torch.optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            ######## 원래는 view -> rehape
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def init_wandb(args):
    
    wandb.init(
        entity=args.wandb_entity, 
        project=args.wandb_project,
        name=args.short,
        config=args,
    )
    wandb.run.save()
    return wandb.config


def save_model(model, optimizer, args, epoch, save_path):
    print(f'==> Saving {save_path}...')
    state = {
        'args': args,
        'online_encoder_projector': model.online_encoder_projector.state_dict(),
        'online_predictor' : model.online_predictor.state_dict(),
        'target_encoder_projector':model.target_encoder_projector.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_path)
    del state




def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
                
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
        warmup_to = eta_min + (args.learning_rate - eta_min) * (
            1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2

        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr















class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_time(elapsed):
    """
    Format time for displaying.
    Arguments:
        elapsed: time interval in seconds.
    """
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))