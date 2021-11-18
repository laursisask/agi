import torch
import torch.nn.functional as F
from torch.nn import Conv2d, Linear, LSTM


class SumoModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear = Linear(7 * 7 * 64, 512)
        self.lstm = LSTM(512, 256, batch_first=True)

        self.value_head = Linear(256, 1)

        # 3 values for forward
        # 3 values for left
        # jumping
        # attacking
        # sprinting
        # delta_yaw mean and standard deviations
        # delta_pitch mean and standard deviations
        self.policy_head = Linear(256, 13)

    def forward(self,
                x,
                initial_state=None,
                compute_value_estimates=True,
                compute_policy_dists=True
                ):
        batch_size = x.size(0)
        sequence_length = x.size(1)

        assert x.shape == (batch_size, sequence_length, 1, 84, 84)

        if not compute_policy_dists and not compute_value_estimates:
            raise ValueError("Should compute at least one of value and policy distributions")

        x = x.reshape([batch_size * sequence_length, 1, 84, 84])

        x = self.conv1(x)
        x = F.relu(x)
        assert x.shape == (batch_size * sequence_length, 32, 20, 20)

        x = self.conv2(x)
        x = F.relu(x)
        assert x.shape == (batch_size * sequence_length, 64, 9, 9)

        x = self.conv3(x)
        x = F.relu(x)
        assert x.shape == (batch_size * sequence_length, 64, 7, 7)

        x = x.flatten(start_dim=1)
        x = self.linear(x)

        assert x.shape == (batch_size * sequence_length, 512)
        x = x.reshape([batch_size, sequence_length, 512])

        if initial_state is not None:
            assert initial_state[0].shape == (1, batch_size, 256)
            assert initial_state[1].shape == (1, batch_size, 256)
            initial_state = (initial_state[0], initial_state[1])

        x, (h_n, c_n) = self.lstm(x, initial_state)

        policy_dists = None
        value = None

        if compute_value_estimates:
            value = self.value_head(x).squeeze(-1)

        if compute_policy_dists:
            policy_dists = self.policy_head(x)

        final_state = torch.stack([h_n, c_n])

        return policy_dists, value, final_state
