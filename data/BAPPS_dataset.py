

import torch
import PIL

from fastai.vision.all import Path

# Omit ScalingLayer
# nomalize [a, b] and then [c, d] is equal to noramlize [a+c*b, b*d]
MEAN = {'BAPPS': [0.485, 0.456, 0.406]}
STD = {'BAPPS': [0.229,0.224,0.225]}

class BAPPS_dataset(torch.utils.data.Dataset):
    def __init__(self, transformation):
        
        root_dir = '/home/mskang/Datasets/PerceptualSimilarity/dataset/2afc/train/'
        self.ref_path = sorted(list(Path(root_dir +  'mix/ref').rglob('*.png')))
        self.transform = transformation

    def __len__(self):
        return len( self.ref_path )   

    def __getitem__(self,idx):   

        ref_image = PIL.Image.open(self.ref_path[idx]).convert('RGB')
        ref_image = self.transform(ref_image)

        return ref_image, torch.empty(ref_image[0].shape[0])
