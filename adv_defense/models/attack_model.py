from torch import nn
import torch


class AttackModel(nn.Module):

    def __init__(self, num_classes):
        super(AttackModel, self).__init__()
        self.inp = nn.Sequential(
            nn.Linear(num_classes, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )

        self.feat = nn.Sequential(
            nn.Linear(num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.out = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        a = self.inp(x)
        b = self.feat(y)
        return self.out(
            torch.cat(
                (a, b),
                dim=1
            )
        )
