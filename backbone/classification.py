import torch.nn as nn
import torch
class ConvNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        img_size = 64
        hidden_dim = int((img_size / 8)**2 * 512)
        self.conv = nn.Sequential(nn.Conv2d(1, 64, (3,3), padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.MaxPool2d((2,2)),
                                   nn.Conv2d(64, 128, (3,3), padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.MaxPool2d((2,2)),
                                   nn.Conv2d(128, 512, (3,3), padding=1),
                                   nn.Conv2d(512, 512, (3,3), padding=1),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(),
                                   nn.MaxPool2d((2,2)),)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 4096),
                                   nn.Linear(4096, 4096),
                                   nn.Linear(4096, n_classes))
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)
