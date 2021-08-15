import torch
import torch.nn.functional as F

from deepq.actions import ACTIONS


class ZombieFightModel(torch.nn.Module):
    def __init__(self):
        super(ZombieFightModel, self).__init__()

        self.conv1_filters = 16
        self.conv2_filters = 32

        # 16 filters, 8x8, stride 4
        self.conv1 = torch.nn.Conv2d(4, self.conv1_filters, kernel_size=8, stride=4)

        # 32 filters, 4x4, stride 2
        self.conv2 = torch.nn.Conv2d(
            self.conv1_filters, self.conv2_filters, kernel_size=4, stride=2
        )

        self.linear_size1 = 256
        self.linear1 = torch.nn.Linear(self.conv2_filters * 9 * 9 + 7, self.linear_size1)

        self.linear2 = torch.nn.Linear(self.linear_size1, len(ACTIONS))

    def forward(self, images, others):
        batch_size = images.size(0)
        assert images.shape == (batch_size, 4, 84, 84)
        assert others.shape == (batch_size, 7)

        x = self.conv1(images)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat([x, others], dim=1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)

        return x
