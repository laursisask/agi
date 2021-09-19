import random
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

import cv2
import torch
import torch.nn.functional as F
from terminator import Terminator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from moving_average_calculator import MovingAverageCalculator
from ppo import action_utils
from ppo.distributions import ModelOutputDistribution
from ppo.model import PPOModel
from processing import create_observation
from stacked_state_constructor import StackedStateConstructor


class EpisodeRecorder:
    def __init__(self, episode):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(f"ppo/videos/episode-{episode}.mp4", fourcc, 20, (336, 336), True)

    def record_state(self, state):
        self.video_writer.write(cv2.cvtColor(state.camera.numpy(), cv2.COLOR_RGB2BGR))

    def finish(self):
        self.video_writer.release()


class Trainer:
    def __init__(self, writer):
        self.writer = writer
        self.batch_size = 32
        self.max_grad_norm = 10.0
        self.num_of_iterations = int(5e6 / 128)
        self.data_collect_steps = 256
        self.num_of_epochs = 3
        self.clip_range = 0.1
        self.stacked_frame_count = 4
        self.model_save_frequency = 25
        self.recording_frequency = 1 / 100
        learning_rate = 1.0e-4

        self.model = PPOModel()
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        argument_parser = ArgumentParser()
        argument_parser.add_argument("--initial-model")
        argument_parser.add_argument("--initial-optimizer")

        args = argument_parser.parse_args()

        if args.initial_model:
            print(f"Loading model weights from {args.initial_model}")
            self.model.load_state_dict(torch.load(args.initial_model))

        if args.initial_optimizer:
            print(f"Loading optimizer parameters from {args.initial_optimizer}")
            self.optimizer.load_state_dict(torch.load(args.initial_optimizers))

        self.model_old = PPOModel()
        self.model_old.train()
        self.model_old.load_state_dict(self.model.state_dict())

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.model_old.to(self.device)

        self.clients = []
        for port in [6660, 6661, 6662]:
            print(f"Connecting to terminator on localhost:{port}")
            c = Terminator()
            c.connect(("localhost", port))
            self.clients.append(c)

        self.executor = ThreadPoolExecutor(max_workers=len(self.clients))

        self.episode_counter = 0
        self.best_model_reward = -1e9
        self.episode_reward_moving_average = MovingAverageCalculator(50)

    def collect_data(self):
        steps_per_client = self.executor.map(self.collect_data_sync, self.clients)

        return [step for steps in steps_per_client for step in steps]

    def collect_data_sync(self, client):
        print("Collecting data")
        steps_done = 0
        steps = []

        while steps_done < self.data_collect_steps:
            # there may be some problems here with concurrent access, like
            # incrementing variables without locks, would be good to fix them
            self.episode_counter += 1
            done = False
            episode_reward = 0
            episode_steps_done = 0
            episode_start = time.time()

            state_constructor = StackedStateConstructor(self.stacked_frame_count)
            state_constructor.append(create_observation(client.reset()))
            initial_raw_state = client.reset()

            record_episode = random.random() < self.recording_frequency

            if record_episode:
                recorder = EpisodeRecorder(self.episode_counter)
                recorder.record_state(initial_raw_state)

            action, state_value = self.pick_action(state_constructor.current())
            state = state_constructor.current()

            while not done:
                steps_done += 1
                episode_steps_done += 1
                next_state, reward, done = client.step(action)

                state_constructor.append(create_observation(next_state))
                next_action, next_state_value = self.pick_action(state_constructor.current())

                advantage = reward + next_state_value - state_value
                steps.append((state, action, reward + next_state_value, advantage))

                action = next_action
                state_value = next_state_value
                state = state_constructor.current()

                episode_reward += reward

                if record_episode:
                    recorder.record_state(next_state)

            print(f"Episode finished with reward {episode_reward:.2f}")
            self.writer.add_scalar("Episode/Reward", episode_reward, self.episode_counter)
            self.episode_reward_moving_average.insert_value(episode_reward)

            episode_end = time.time()
            episode_duration = episode_end - episode_start
            tps = episode_steps_done / episode_duration
            print(f"Episode took {episode_steps_done} steps and {episode_duration:.2f} seconds, TPS was {tps:.2f}")
            self.writer.add_scalar("Episode/Length", episode_steps_done, self.episode_counter)
            writer.add_scalar("Episode/TPS", tps, self.episode_counter)

            if record_episode:
                recorder.finish()

        print("Finished collecting data")

        return steps

    @torch.inference_mode()
    def pick_action(self, state):
        images, others = state

        images = images.to(self.device)
        others = others.to(self.device)

        output = self.model(images=images.unsqueeze(0), others=others.unsqueeze(0))

        dist = ModelOutputDistribution(output)

        action = action_utils.to_python_types(dist.sample())
        value = output.value.item()

        assert action.forward in [-1, 0, 1]
        assert action.left in [-1, 0, 1]
        assert type(action.jumping) == bool
        assert type(action.attacking) == bool
        assert type(action.delta_pitch) == float
        assert type(action.delta_yaw) == float
        assert type(value) == float

        # check that values do not explode
        assert -500 < action.delta_pitch < 500
        assert -500 < action.delta_yaw < 500
        assert -100 < value < 100

        return action, value

    def train(self):
        print("Optimizing parameters")
        global_iteration = 0

        for iteration in range(self.num_of_iterations):
            steps = self.collect_data()

            if self.episode_reward_moving_average.get() > self.best_model_reward:
                print(f"Current model has higher reward moving average ({self.episode_reward_moving_average.get():.2f})"
                      f" compared to previous one ({self.best_model_reward:.2f}), saving model")

                torch.save(self.model.state_dict(), "best_model.pt")

                self.best_model_reward = self.episode_reward_moving_average.get()

            data_loader = DataLoader(steps, batch_size=self.batch_size, shuffle=True, drop_last=True)

            for epoch in range(self.num_of_epochs):
                for (images, others), actions, target_state_values, advantages in data_loader:
                    batch_size = advantages.size(0)
                    assert batch_size > 1

                    global_iteration += 1
                    self.optimizer.zero_grad()

                    images = images.to(self.device)
                    others = others.to(self.device)
                    actions = action_utils.to_device(actions, self.device)
                    target_state_values = target_state_values.to(torch.float32).to(self.device)
                    advantages = advantages.to(torch.float32).to(self.device)

                    output = self.model(images=images, others=others)

                    standardised_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

                    with torch.no_grad():
                        output_old = self.model_old(images=images, others=others)

                    action_dist = ModelOutputDistribution(output)
                    action_dist_old = ModelOutputDistribution(output_old)

                    ratios = torch.exp(action_dist.log_prob(actions) - action_dist_old.log_prob(actions))

                    objective = torch.min(
                        ratios * standardised_advantages,
                        torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * standardised_advantages
                    )

                    actor_loss = -objective.mean()
                    critic_loss = F.mse_loss(output.value, target_state_values)
                    entropy = action_dist.entropy()

                    assert not actor_loss.isnan(), "Actor loss was NaN"
                    assert not critic_loss.isnan(), "Critic loss was NaN"
                    assert not entropy.isnan(), "Entropy was NaN"

                    assert not actor_loss.isinf(), "Actor loss was infinite"
                    assert not critic_loss.isinf(), "Critic loss was infinite"
                    assert not entropy.isinf(), "Entropy was infinite"

                    total_loss = actor_loss + critic_loss - 0.05 * entropy

                    total_loss.backward()

                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()

                    self.writer.add_scalar("Unclipped gradient norm", grad_norm.item(), global_iteration)
                    self.writer.add_scalar("Loss/Total", total_loss.item(), global_iteration)
                    self.writer.add_scalar("Loss/Actor", actor_loss.item(), global_iteration)
                    self.writer.add_scalar("Loss/Critic", critic_loss.item(), global_iteration)
                    self.writer.add_scalar("Loss/Entropy", entropy.item(), global_iteration)

            self.model_old.load_state_dict(self.model.state_dict())

            if iteration % self.model_save_frequency == 1:
                print("Saving model")
                torch.save(self.model.state_dict(), "last_model.pt")


if __name__ == "__main__":
    with SummaryWriter() as writer:
        Trainer(writer).train()
