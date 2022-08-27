from torchvision import transforms, datasets
import torch
from .utils import TwoCropTransform

def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'BAPPS':
        mean = (0.485, 0.456, 0.406)
        std = (0.229,0.224,0.225)
    
    normalize = transforms.Normalize(mean=mean, std=std)

    if opt.distortion == 'supcon':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

    elif opt.distortion == 'rand':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandAugment(2, 9),
            transforms.ToTensor(),
            normalize,
        ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./datasets',
                                        transform=TwoCropTransform(train_transform),
                                        download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./datasets',
                                        transform=TwoCropTransform(train_transform),
                                        download=True)
    elif opt.dataset == 'BAPPS':
        from .BAPPS_dataset import BAPPS_dataset
        train_dataset = BAPPS_dataset(transformation = TwoCropTransform(train_transform))

    train_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader

