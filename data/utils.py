from PIL import ImageFilter, ImageOps
import random
import torchvision.transforms.functional as tf
'''
Refer
https://github.com/facebookresearch/moco-v3/blob/main/moco/loader.py
https://github.com/facebookresearch/moco/blob/main/moco/loader.py
'''

class TwoCropTransform:
    '''https://github.com/kiimmm/GenSCL/blob/main/data/utils.py'''
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)