

import torch
import PIL

# from fastai.vision.all import Path
import os

# Omit ScalingLayer
# nomalize [a, b] and then [c, d] is equal to noramlize [a+c*b, b*d]
MEAN = {'BAPPS': [0.485, 0.456, 0.406]}
STD = {'BAPPS': [0.229,0.224,0.225]}

class BAPPS_dataset(torch.utils.data.Dataset):
    def __init__(self, transformation):
        
        root_dir = '/home/mskang/SimPS/dataset/2afc/train/mix/ref'
        self.ref_path = [root_dir+'/'+file for file in os.listdir(root_dir) if file.endswith('.png')]
        # self.ref_path = sorted(list(Path(root_dir).rglob('*.png')))
        self.transform = transformation

    def __len__(self):
        return len( self.ref_path )   

    def __getitem__(self,idx):   

        ref_image = PIL.Image.open(self.ref_path[idx])
        ref_image = ref_image.convert('RGB')
        ref_image = self.transform(ref_image)

        return ref_image, torch.empty(1)
