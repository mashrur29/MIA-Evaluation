from torch import nn
import torch
import torch.nn.functional as F

class RNNClassifier(nn.Module):
    def __init__(self, inp_size, num_classes):
        super(RNNClassifier, self).__init__()

        self.hidden = nn.Sequential(
            nn.Linear(inp_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.out = nn.RNN(128, num_classes)
        self.h0 = torch.randn(1, num_classes).cuda()

    def forward(self, x):
        h = self.hidden(x)
        out_tensor = self.out(h, self.h0)
        return F.relu(out_tensor[0]), out_tensor[1]
