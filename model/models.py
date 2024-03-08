import torch
import torch.nn.functional as F
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, batch_norm:bool, activation:str):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size, input_size)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.batch_norm = batch_norm
        self.activation = ACTIVATION[activation]
        
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.bn2 = nn.BatchNorm1d(input_size)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.fc2(out)
        if self.batch_norm:
            out = self.bn2(out)
        out += residual  # Skip connection
        out = self.activation(out)
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
        self.activation=activation
        self.lr=lr

        all_layers=[]
        for hidden_unit in self.block_pattern: 
            res_block = ResidualBlock(self.input_size, hidden_unit, self.batch_norm, self.activation)
            all_layers.append(res_block)
            self.input_size=hidden_unit
        last_layer=nn.Linear(hidden_unit, self.output_size)
        nn.init.kaiming_normal_(last_layer.weight)
        all_layers.append(last_layer)
        self.block=nn.Sequential(*all_layers)

    def forward(self, x): 
        out=self.block(x)
        return out