import torch.nn as nn

class LinearClassifier(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
