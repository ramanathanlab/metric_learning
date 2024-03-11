import torch
import torch.nn.functional as F
import torch.nn as nn
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
import matplotlib.pyplot as plt
# constants
ACTIVATION={'ReLU':nn.ReLU()}
OPTIMIZERS={'AdamW':AdamW, 
            'CosineAnnealing':CosineAnnealingLR}
MINERS={'None':miners.EmbeddingsAlreadyPackagedAsTriplets()}
LOSS={'Contrastive':losses.ContrastiveLoss,
        'NTXent':losses.NTXentLoss,
        'NCA':losses.NCALoss}
ACCURACY={'Standard':AccuracyCalculator}


class DropoutLayer(nn.Module):
    def __init__(self, p=0.5):
        super(DropoutLayer, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:  # Only apply dropout during training
            batch_size, num_features = x.shape
            dropout_mask = torch.rand(batch_size, num_features) < self.p
            while dropout_mask.all(dim=1).any():  # Check if any row in the mask is all False (no dropout applied)
                # Re-generate mask for rows with no dropout
                rows_to_adjust = ~dropout_mask.all(dim=1)
                dropout_mask[rows_to_adjust] = torch.rand(rows_to_adjust.sum(), num_features) < self.p
            return x * dropout_mask.type_as(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, input_size:int, block_pattern:list,
                 output_size:int, batch_norm:bool, activation:str):
        super(ResidualBlock, self).__init__()
        self.input_size=input_size
        self.block_pattern=block_pattern
        self.output_size=output_size
        self.activation=activation
        self.batch_norm=batch_norm
        self.drop=DropoutLayer(p=0.3) # set to half 
        self.block_pattern=block_pattern

        all_layers=[]
        for hidden_unit in self.block_pattern:
            layer=nn.Linear(self.input_size, hidden_unit)
            nn.init.kaiming_normal_(layer.weight)
            all_layers.append(layer)
            all_layers.append(self.activation)
            if self.batch_norm: 
                bn=nn.BatchNorm1d(hidden_unit)
                all_layers.append(bn)
            drop=DropoutLayer(p=0.3) # set to 30% 
            all_layers.append(drop)
            self.input_size=hidden_unit
        last_layer=nn.Linear(hidden_unit, self.output_size)
        nn.init.kaiming_normal_(last_layer.weight)
        all_layers.append(last_layer)
        self.block=nn.Sequential(*all_layers)
        
    def forward(self, x): 
        out = self.block(x)
        out += x
        out = self.activation(x)
        return out

class MetricNet(nn.Module): 
    def __init__(self,
                 num_blocks:int,
                 input_size:int, 
                 output_size:int, 
                 block_pattern:list,
                 batch_norm:bool,
                 activation:str,
                 lr:float
                 ): 
        super(MetricNet, self).__init__()
        self.num_blocks=num_blocks
        self.input_size=input_size
        self.output_size=output_size
        self.block_pattern=block_pattern
        self.batch_norm=batch_norm
        self.activation=ACTIVATION[activation]

        self.lr=lr
        self.blocks = nn.Sequential(*[ResidualBlock(self.input_size, self.block_pattern, self.output_size, 
                                    self.batch_norm, self.activation)]*self.num_blocks)

    def forward(self, x): 
        out=self.blocks(x)
        return out
    