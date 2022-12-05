from torch import nn
import torch
import torch.nn.functional as F

class RNNClassifier(nn.Module):
    def __init__(self, inp_size, num_classes):
        super(RNNClassifier, self).__init__()

        self.hidden = nn.Sequential(
            nn.Linear(inp_size, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        )
        self.out = nn.RNN(128, num_classes)
        self.h0 = torch.randn(1, num_classes).cuda()

    def forward(self, x):
        h = self.hidden(x)
        out_tensor = self.out(h, self.h0)
        return F.tanh(out_tensor[0]), out_tensor[1]
