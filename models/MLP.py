import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, mid_dim, prediction_dim): # Note that dimension of prediction and projector should be same
        super(MLP, self).__init__()

        self.predictor = nn.Sequential(
                                       nn.Linear(in_dim, mid_dim),
                                       nn.BatchNorm1d(mid_dim),
                                       nn.ReLU(inplace = True),
                                       nn.Linear(mid_dim, prediction_dim)
                                       )
    def forward(self, x):
        return self.predictor(x)