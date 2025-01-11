import torch
import torch.nn as nn
import torch.nn.functional as F


class STN3D(nn.Module):
    def __init__(self, input_channels=3):
        super(STN3D, self).__init__()
        self.input_channels = input_channels
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_channels * input_channels),
        )

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = self.mlp1(x)
        x = F.max_pool1d(x, num_points).squeeze(2)
        x = self.mlp2(x)
        I = torch.eye(self.input_channels).view(-1).to(x.device)
        x = x + I
        x = x.view(-1, self.input_channels, self.input_channels)
        return x


class TargetEncoder(nn.Module):
    def __init__(self, embedding_size=256, input_channels=3, is_src=False, sem_size=False):
        super(TargetEncoder, self).__init__()
        self.input_channels = input_channels
        self.is_src = is_src
        self.max_part = 16
        self.sem_size = sem_size
        self.stn1 = STN3D(input_channels)
        self.stn2 = STN3D(64)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        if self.sem_size != False:
            self.fuse_sem = nn.Sequential(
                nn.Conv1d(1024 + sem_size, 1024, 1),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                # nn.Conv1d(1024, 1024, 1),
                # nn.BatchNorm1d(1024),
                # nn.ReLU(),
            )
        self.per_point_out = nn.Sequential(
            nn.Conv1d(1024, embedding_size, 1),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(),
            nn.Conv1d(embedding_size, embedding_size, 1)
        )
        self.fc = nn.Linear(1024, embedding_size)

    def forward(self, x, sem_f):
        batch_size = x.shape[0]
        num_points = x.shape[-2]
        if self.is_src:
            max_part = x.shape[1]
            x = x.view(batch_size * max_part, num_points, self.input_channels)
        x = x.transpose(2, 1)  # transpose to apply 1D convolution
        x = self.mlp1(x)
        x = self.mlp2(x)
        if self.sem_size != False:
            if self.is_src:
                sem_f = sem_f.view(batch_size * max_part, -1, 1).repeat(1, 1, num_points)
            else:
                sem_f = sem_f.transpose(2, 1)
            x = torch.cat([x, sem_f], dim=1)
            x = self.fuse_sem(x)
        per_point = self.per_point_out(x)
        x = F.max_pool1d(x, num_points).squeeze(2)  # max pooling
        x = self.fc(x)
        return x, per_point # x:[bs, c], per_point:[bs, c, n]


class SrcEncoder(nn.Module):
    def __init__(self, embedding_size=256, input_channels=3):
        super(SrcEncoder, self).__init__()
        self.input_channels = input_channels
        self.stn1 = STN3D(input_channels)
        self.stn2 = STN3D(64)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.fc = nn.Linear(1024, embedding_size)

    def forward(self, x):
        batch_size = x.shape[0]
        num_points = x.shape[1]
        x = x[:, :, : self.input_channels]
        x = x.transpose(2, 1)  # transpose to apply 1D convolution
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = F.max_pool1d(x, num_points).squeeze(2)  # max pooling
        x = self.fc(x)
        return x
