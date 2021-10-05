import datetime
import logging
import os
import random
import tempfile
from argparse import ArgumentParser

import cv2
import gym
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.nn import Conv2d, Linear, LSTM
from torch.utils.tensorboard import SummaryWriter

from dppo.dppo import configure_logger, collect_data_and_train


def transform_raw_state(state):
    assert (210, 160, 3) == state.shape

    permuted_image = torch.tensor(state).permute(2, 0, 1)
    assert permuted_image.shape == (3, 210, 160)

    grayscaled_image = TF.rgb_to_grayscale(permuted_image)
    assert grayscaled_image.shape == (1, 210, 160)

    resized_image = TF.resize(grayscaled_image, [110, 84])
    assert resized_image.shape == (1, 110, 84)

    cropped_image = TF.crop(resized_image, top=18, left=0, height=84, width=84)
    assert cropped_image.shape == (1, 84, 84)

    normalized_image = TF.normalize(cropped_image.float(), [128], [256])

    return normalized_image


class PongEnvWrapper:
    def __init__(self):
        self.env = gym.make("Pong-v0")
        self.record_episode = False
        self.recorder = None

    def reset(self):
        if self.record_episode and self.recorder.isOpened():
            self.recorder.release()

        self.record_episode = random.random() < 0.01

        raw_observation = self.env.reset()

        if self.record_episode:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            file = f"videos/{datetime.datetime.now().isoformat()}.mp4"
            os.makedirs(os.path.dirname(file), exist_ok=True)
            self.recorder = cv2.VideoWriter(file, fourcc, 30, (160, 210), True)
            self.recorder.write(cv2.cvtColor(raw_observation, cv2.COLOR_RGB2BGR))

        return transform_raw_state(raw_observation)

    def step(self, action):
        raw_observation, reward, done, info = self.env.step(action)

        if self.record_episode:
            self.recorder.write(cv2.cvtColor(raw_observation, cv2.COLOR_RGB2BGR))

            if done:
                self.recorder.release()

        return transform_raw_state(raw_observation), reward, done, info


class PongModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear = Linear(7 * 7 * 64, 512)
        self.lstm = LSTM(512, 256, batch_first=True)

        self.value_head = Linear(256, 1)
        self.policy_head = Linear(256, 6)

    def forward(self, x, initial_state=None, compute_value_estimates=True, compute_policy_dists=True):
        batch_size = x.size(0)
        sequence_length = x.size(1)

        assert x.shape == (batch_size, sequence_length, 1, 84, 84)

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
            h_0, c_0 = initial_state
            initial_state = (h_0, c_0)

        x, (h_n, c_n) = self.lstm(x, initial_state)

        if compute_value_estimates:
            value = self.value_head(x).squeeze(-1)

        if compute_policy_dists:
            policy_dists = F.log_softmax(self.policy_head(x), dim=-1)

        final_state = torch.stack([h_n, c_n])

        if compute_value_estimates and compute_policy_dists:
            return policy_dists, value, final_state
        elif compute_policy_dists:
            return policy_dists, final_state
        elif compute_value_estimates:
            return value, final_state
        else:
            raise ValueError("Should compute at least one of value and policy distributions")


class PolicyState:
    def __init__(self, model, device, observation):
        self.model = model
        self.device = device
        self.last_state = None
        self.policy_dist = None

        self.update(observation)

    @torch.no_grad()
    def update(self, observation):
        policy_dists, self.last_state = self.model(
            observation.unsqueeze(0).unsqueeze(0).to(self.device),
            initial_state=self.last_state,
            compute_value_estimates=False
        )

        self.policy_dist = torch.exp(policy_dists.squeeze(0).squeeze(0))

    @torch.no_grad()
    def sample_action(self):
        return torch.multinomial(self.policy_dist, num_samples=1)


class Policy:
    def __init__(self, model, device):
        super().__init__()

        self.model = model
        self.device = device

    def action_log_prob(self, policy_dists, actions):
        return torch.gather(policy_dists, index=actions, dim=-1).squeeze(-1)

    def get_initial_state(self, observation):
        return PolicyState(self.model, self.device, observation)

    def entropy(self, policy_dists):
        min_real = torch.finfo(policy_dists.dtype).min
        logits = torch.clamp(policy_dists, min=min_real)
        p_log_p = logits * F.softmax(logits, dim=-1)
        return -p_log_p.sum(dim=-1)


def launch_worker(rank, world_size, tmp_path, initial_model, initial_optimizer):
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{tmp_path}/sharedfile",
        rank=rank,
        world_size=world_size
    )

    configure_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using {device} device")

    model = PongModel()
    if initial_model is not None:
        logging.info(f"Loading model weights from {initial_model}")
        model.load_state_dict(torch.load(initial_model, map_location=torch.device("cpu")))

    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
    if initial_optimizer is not None:
        logging.info(f"Loading optimizer weights from {initial_optimizer}")
        state_dict = torch.load(initial_optimizer)
        optimizer.load_state_dict(state_dict)

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    model.to(device)

    eval_reward = collect_data_and_train(
        device=device,
        policy=Policy(model, device),
        envs=[PongEnvWrapper() for _ in range(8)],
        model=model,
        optimizer=optimizer,
        metrics=SummaryWriter(),
        iterations=10000,
        episodes_per_iteration=64,
        bootstrap_length=5,
        num_epochs=3,
        batch_size=8,
        backpropagation_steps=10,
        clip_range=0.1,
        max_grad_norm=5.0,
        value_coefficient=1.0,
        entropy_coefficient=0.1,
        evaluation_episodes=256
    )

    if rank == 0:
        print(eval_reward)


def main():
    nprocs = 1
    tmp_path = tempfile.mkdtemp()

    parser = ArgumentParser()
    parser.add_argument("--initial-model", required=False)
    parser.add_argument("--initial-optimizer", required=False)

    args = parser.parse_args()

    mp.spawn(launch_worker, args=(nprocs, tmp_path, args.initial_model, args.initial_optimizer), nprocs=nprocs)


if __name__ == "__main__":
    main()
