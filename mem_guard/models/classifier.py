from torch import nn


class Classifier(nn.Module):
    def __init__(self, inp_size, num_classes):
        super(Classifier, self).__init__()

        self.hidden = nn.Sequential (
            nn.Linear(inp_size, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        )

        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        h = self.hidden(x)
        out_tensor = self.out(h)
        return out_tensor