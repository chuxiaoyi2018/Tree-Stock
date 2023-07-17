import torch
import torch.nn as nn
import torch.nn.functional as F


class StockDNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size = [1024, 512, 256]) -> None:
        super(StockDNN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.loss_func = nn.CrossEntropyLoss()
        
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size[0]),
#             nn.Dropout(0.5),
#             nn.BatchNorm1d(hidden_size[0]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size[0], hidden_size[1]),
#             nn.Dropout(0.5),
#             nn.BatchNorm1d(hidden_size[1]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size[1], hidden_size[2]),
#             nn.Dropout(0.5),
#             nn.BatchNorm1d(hidden_size[2]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size[2], self.output_dim),
        )
    
    def forward(self, x):
        output = self.mlp(x)
        return output
            
    def compute_loss(self, x, target):
        output = self.forward(x)
        loss = self.loss_func(output, target.long())
        return loss