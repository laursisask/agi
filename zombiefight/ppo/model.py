from collections import namedtuple

import torch
import torch.nn.functional as F

ContinuousActionOutput = namedtuple("ContinuousActionOutput", ["means", "stds"])


class ContinuousActionHead(torch.nn.Module):
    def __init__(self, in_features, num_of_actions):
        super(ContinuousActionHead, self).__init__()

        self.mean_layer = torch.nn.Linear(in_features, num_of_actions)
        self.std_layer = torch.nn.Linear(in_features, num_of_actions)

    def forward(self, x):
        means = (torch.sigmoid(self.mean_layer(x)) - 0.5) * 12
        stds = torch.sigmoid(self.std_layer(x)) * 6

        return ContinuousActionOutput(means=means, stds=stds)


ModelOutput = namedtuple("ModelOutput", ["mouse", "forward_movement", "strafe_movement", "jump", "attack", "value"])


class PPOModel(torch.nn.Module):
    def __init__(self):
        super(PPOModel, self).__init__()

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

        # One action for pitch, one for yaw
        self.mouse_head = ContinuousActionHead(self.linear_size1, 2)

        # go forward, do nothing, go backwards
        self.forward_movement_head = torch.nn.Linear(self.linear_size1, 3)

        # go left, do nothing, go right
        self.strafe_movement_head = torch.nn.Linear(self.linear_size1, 3)

        self.jump_head = torch.nn.Linear(self.linear_size1, 1)
        self.attack_head = torch.nn.Linear(self.linear_size1, 1)

        self.value_head = torch.nn.Linear(self.linear_size1, 1)

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

        assert x.shape == (batch_size, self.linear_size1)

        mouse = self.mouse_head(x)
        forward_movement = self.forward_movement_head(x)
        strafe_movement = self.strafe_movement_head(x)
        jump = self.jump_head(x).squeeze(1)
        attack = self.attack_head(x).squeeze(1)
        value = torch.sigmoid(self.value_head(x).squeeze(1)) * 12

        assert mouse.means.shape == (batch_size, 2)
        assert mouse.stds.shape == (batch_size, 2)
        assert forward_movement.shape == (batch_size, 3)
        assert strafe_movement.shape == (batch_size, 3)
        assert jump.shape == (batch_size,)
        assert attack.shape == (batch_size,)
        assert value.shape == (batch_size,)

        return ModelOutput(
            mouse=mouse,
            forward_movement=forward_movement,
            strafe_movement=strafe_movement,
            jump=jump,
            attack=attack,
            value=value
        )
