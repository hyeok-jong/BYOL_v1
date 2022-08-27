import torch
import torch.nn as nn
import torchvision.models as ms
import copy 
import time
import datetime
import wandb
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn

from utils import set_dir, save_model, format_time
from utils import init_wandb
from utils import adjust_learning_rate, warmup_learning_rate, get_learning_rate
from parser import BYOL_parser
from data import set_loader


def set_model(name, representation_dim = 128):
    models = {
              'alex' : ms.alexnet(weights = None, num_classes = representation_dim),
              'resnet18' : ms.resnet18(weights = None, num_classes = representation_dim),
              'resnet34' : ms.resnet34(weights = None, num_classes = representation_dim),
              'resnet50' : ms.resnet50(weights = None, num_classes = representation_dim),
              'vgg16' : ms.vgg16(weights = None, num_classes = representation_dim),
              'vgg19' : ms.vgg19(weights = None, num_classes = representation_dim)}

    torch.backends.cudnn.benchmark = True
    return models[name]

def set_predictor(dim = 128, projection_size = 128, hidden_size=128):
    # Note that size(dim) of projection and prediction should be same
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )


def loss_function(online_prediction, target_projection):
    x = torch.nn.functional.normalize(online_prediction, dim=-1, p=2)
    y = torch.nn.functional.normalize(target_projection, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def stop_gradient(model):
    for p in model.parameters():
        p.requires_grad = False

class exponential_moving_average():
    def __init__(self, beta = 0.99):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_target_encoder_projector(ema_updater, target, online):
    for online_params, target_params in zip(online.parameters(), target.parameters()):
        old_weight, up_weight = target_params.data, online_params.data
        target_params.data = ema_updater.update_average(old_weight, up_weight)


class BYOL(torch.nn.Module):
    # Class for BYOL 
    # It's for update target network
    def __init__(self, args):
        super().__init__()

        self.online_encoder_projector = set_model(args.model, args.represent_dim)
        self.target_encoder_projector = copy.deepcopy(self.online_encoder_projector)
        stop_gradient(self.target_encoder_projector)
        self.online_predictor = set_predictor(args.dim, args.projection_size, hidden_size=4096)

        self.target_ema_updater = exponential_moving_average(0.99)
        self.device = args.device

    def update_target(self):
        update_target_encoder_projector(self.target_ema_updater, self.target_encoder_projector, self.online_encoder_projector)

    def forward(self, images):
        image1 = images[0].to(self.device)
        image2 = images[1].to(self.device)

        online_projection_1 = self.online_encoder_projector(image1)
        online_projection_2 = self.online_encoder_projector(image2)    

        online_prediction_1 = self.online_predictor(online_projection_1)
        online_prediction_2 = self.online_predictor(online_projection_2)

        with torch.no_grad():
            #self.target_encoder_projector = self.set_target_encoder_projector()
            target_projection_1 = self.target_encoder_projector(image1)
            target_projection_2 = self.target_encoder_projector(image2)
        
        loss_1 = loss_function(online_prediction_1, target_projection_1)
        loss_2 = loss_function(online_prediction_2, target_projection_2)

        loss = loss_1 + loss_2

        return loss.mean()

def train():

    set_dir()
    args = BYOL_parser()
    init_wandb(args)

    BYOL_model = BYOL(args)
    BYOL_model.to(args.device)
    BYOL_model.train()
    print(BYOL_model)
    for name,i in BYOL_model.named_parameters():
        print(name, i.requires_grad)
    train_loader = set_loader(args)

    optimizer = torch.optim.Adam(BYOL_model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.Adam(list(BYOL_model.online_encoder_projector.parameters())+ list(BYOL_model.online_predictor.parameters()), lr=args.learning_rate)
    cudnn.bechmark = True

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)
        training_loss = list()
        start_time = time.time()

        # For 1-epoch training

        for idx, (images, _) in enumerate(train_loader):
            warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)
            loss = BYOL_model( images )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            BYOL_model.update_target()

            training_loss.append(loss.cpu().detach().item())

        loss_epoch = np.mean(training_loss)
        lr = get_learning_rate(optimizer)
        print(f'[epoch:{epoch}/{args.epochs}] [loss:{np.round(loss_epoch,6)}] [Time:[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]] [total time:{format_time(time.time() - start_time)}][lr:{np.round(lr,6)}]')
        res = {'training loss' : loss_epoch, 'learning rate' : lr}

        wandb.log(res)

        # Save logs
        path_to_save = f'./result/{args.distortion}/{args.dataset}/{args.model}'
        if (epoch % args.save_freq == 0) or (epoch<6) or ((epoch == args.epochs)): 
            save_model(BYOL_model, optimizer, args, epoch, save_path = f'{path_to_save}/{epoch}.pt')
            
    wandb.finish()

if __name__ == '__main__':
    train()