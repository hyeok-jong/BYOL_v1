from models import resnet_BYOL, MLP

import torch
import torch.backends.cudnn as cudnn 
import torch.nn.functional as tnf

import numpy as np

import time
import datetime

import wandb

from data import set_loader

from utils import warmup_learning_rate, get_learning_rate, AverageMeter
from utils import init_wandb, set_dir, format_time, save_logs
from utils import adjust_learning_rate
from parser import BYOL_parser



def set_encoder_projector(args):
    # Other models will be able soon
    if args.model[:3] == 'res':
        model = resnet_BYOL(
                            model_name = args.model,
                            mid_dim = args.mid_dim,
                            projection_dim = args.projection_dim,
                            mode = 'BYOL'
                            ).to(args.device)
        projection_dim = model.projection_dim
    cudnn.benchmark = True
    return model, projection_dim

def loss_function(x, y):
    x = tnf.normalize(x, dim = 1)
    y = tnf.normalize(y, dim = 1)
    return 2 - 2 * (x * y).sum(dim = -1)

def initialize_target(online, target):
    for online_params, target_params in zip(online.parameters(), target.parameters()):
        target_params.data.copy_(online_params)
        target_params.requires_grad = False

def update_target(online, target, moment = 0.99):
    for online_params, target_params in zip(online.parameters(), target.parameters()):
        target_params.data = target_params.data * moment + online_params.data * (1. - moment)
        # if not use .data "RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation.""


def train(train_loader, online_encoder_projector, online_predictor, target_encoder_projector, optimizer, epoch, args):
    online_encoder_projector.train()
    online_predictor.train()
    target_encoder_projector.train()
    losses = AverageMeter()

    ''' 1 - epoch training '''
    for idx, (images, _) in enumerate(train_loader):
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)
        image1 = images[0].to(args.device)
        image2 = images[1].to(args.device)

        batch_size = image1.shape[0]

        online_projection1 = online_encoder_projector(image1)
        online_projection2 = online_encoder_projector(image2)

        online_prediction1 = online_predictor(online_projection1)
        online_prediction2 = online_predictor(online_projection2)

        with torch.no_grad():
            target_projection1 = target_encoder_projector(image1)
            target_projection2 = target_encoder_projector(image2)


        loss1 = loss_function(online_prediction1, target_projection1)
        loss2 = loss_function(online_prediction2, target_projection2)

        loss = (loss1 + loss2).mean()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_target(online_encoder_projector, target_encoder_projector)

        losses.update(loss.item(), batch_size)

    res = { 'training_loss' : losses.avg, 
            'learning_rate' : get_learning_rate(optimizer) }
    return res

def main():
    set_dir()
    args = BYOL_parser()

    init_wandb(args)

    train_loader = set_loader(args)

    online_encoder_projector, projection_dim = set_encoder_projector(args)
    online_predictor = MLP(projection_dim, args.mid_dim, args.prediction_dim)
    target_encoder_projector, _ = set_encoder_projector(args)

    initialize_target(online_encoder_projector, target_encoder_projector)

    '''    
    optimizer = torch.optim.SGD(list(online_encoder_projector.parameters())+list(online_predictor.parameters()),
                                lr = args.learning_rate,
                                momentum = args.momentum,
                                weight_decay = args.weight_decay)
    '''
    optimizer = torch.optim.Adam(list(online_encoder_projector.parameters())+list(online_predictor.parameters()),
                                lr = args.learning_rate)


    online_encoder_projector.to(args.device)
    online_predictor.to(args.device)
    target_encoder_projector.to(args.device)


    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)

        start_time = time.time()
        res = train(train_loader, online_encoder_projector, online_predictor, target_encoder_projector, optimizer, epoch, args)
        loss = res['training_loss']
        lr = res['learning_rate']
        print(f'[epoch:{epoch}/{args.epochs}] [loss:{loss}] [lr:{np.round(lr,6)}] [Time:[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]] [total time:{format_time(time.time() - start_time)}]')

        wandb.log(res, step = epoch)
        save_logs(args, epoch, online_encoder_projector, optimizer, loss, lr)
        
    wandb.finish()

if __name__ == '__main__':
    main()