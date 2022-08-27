import torch.nn as nn
import torchvision.models as tm

from .MLP import MLP
from .identity import Identity


class resnet_BYOL(nn.Module):
    def __init__(self, model_name, mid_dim = 100, projection_dim = 100, mode = 'BYOL'):
        super(resnet_BYOL, self).__init__()

        self.encoder, in_dim  = self.get_encoder(model_name)

        if mode == "BYOL":
            self.projector = MLP(in_dim, mid_dim, projection_dim)
        elif mode == "Linear":
            self.projector = nn.Linear(in_dim, 10)

        self.projection_dim = projection_dim

    def get_encoder(self, model_name):

        resnet_dict = {"resnet18": tm.resnet18(weights = None, num_classes = 10),
                       "resnet34": tm.resnet34(weights = None, num_classes = 10),
                       "resnet50": tm.resnet50(weights = None, num_classes = 10)}

        model = resnet_dict[model_name]
        dim_in = model.fc.in_features
        model.fc = Identity()
        return model, dim_in

    def forward(self, x):
        x = self.encoder(x)
        x = self.projector(x)
        return x
