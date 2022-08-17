import argparse
import datetime

def BYOL_parser():
    parser = argparse.ArgumentParser('argument for training')

    # Save
    parser.add_argument('--save_freq', type = int, default = 15,
                        help = 'save frequency')
    
    # Training
    parser.add_argument('--batch_size', type = int, default = 1024,
                        help = 'batch_size')
    parser.add_argument('--num_workers', type = int, default = 8,
                        help = 'num of workers to use')
    parser.add_argument('--epochs', type = int, default = 200,
                        help = 'number of training epochs')

    # Optimization
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--learning_rate', type = float, default = 0.01,
                        help = 'learning rate')
    parser.add_argument('--momentum', type = float, default = 0.9,
                        help = 'momentum')
    parser.add_argument('--cosine', action = 'store_true',
                        help = 'using cosine annealing')
    parser.add_argument('--warm', action = 'store_true',
                        help = 'warm-up for large batch training')
    parser.add_argument('--warmup-from', type=float, default=0.01)
    parser.add_argument('--warm-epochs', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--lr_decay_epochs', type=int, nargs='+', default=[600, 800, 900],
                        help='where to decay lr, can be a list')

    # Model & Dataset
    parser.add_argument('--model', type = str, default = 'resnet18')
    parser.add_argument('--dataset', type = str, default = 'cifar10',
                        choices = ['cifar10', 'cifar100', 'BAPPS', 'stl10'], help = 'dataset')
    parser.add_argument('--size', type = int, default = 64, help = 'parameter for RandomResizedCrop')
    parser.add_argument('--represent_dim', type = int, default = 1024)
    parser.add_argument('--dim', type = int, default = 1024)
    parser.add_argument('--projection_size', type = int, default = 1024)
    # Custom
    parser.add_argument('--distortion', type = str, default = 'supcon',
                        help = '[supcon / rand / v1 / v2 / v3]')
    parser.add_argument('--device', type = str, default = 'cuda:1')

    # wandb
    '''
    parser.add_argument('--wandb', action='store_true', default=True,
                        help='use wandb for visualization')
    '''
    parser.add_argument('--wandb_entity', type=str, default='hyeokjong',
                        help='Wandb ID')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Project name')
    parser.add_argument('--short', type=str, default=None,
                        help='short name')
    args = parser.parse_args()
    
    # Linear evaluation
    parser.add_argument('--pt_path', type = str, default='',
                        help='use wandb for visualization')

    if args.wandb_project == None:
        args.wandb_project = f'[BYOL][{args.distortion}][{args.dataset}][{args.model}]'
    if args.short == None:
        args.short = f'[Training][batch:{args.batch_size}][input:{args.size}][{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]'

    return args