import argparse
import itertools
import os
import time
import random
from threading import Thread

import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import cv2
from terminator import Terminator

from actions import ACTIONS
from model import ZombieFightModel
from processing import create_observation
from stacked_state_constructor import StackedStateConstructor

ports = [6660, 6661]
clients = []
for port in ports:
    c = Terminator()
    c.connect(("localhost", port))
    print(f"Connected to MC client on port {port}")
    clients.append(c)


class EpisodeRecorder:
    def __init__(self, episode):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(f"videos/episode-{episode}.mp4", fourcc, 20, (84, 84), False)

    def record_state(self, state):
        assert state.shape == (84, 84)

        self.video_writer.write((state * 256 + 128).to(dtype=torch.uint8).reshape([84, 84, 1]).numpy())

    def finish(self):
        self.video_writer.release()


class MovingAverageCalculator:
    def __init__(self, k):
        self.k = k
        self.items = []
        self.current_sum = 0

    def insert_value(self, value):
        self.items.append(value)
        self.current_sum += value

        if len(self.items) > self.k:
            self.current_sum -= self.items[0]
            del self.items[0]

    def get(self):
        return self.current_sum / len(self.items)


class ReplayMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def append(self, item):
        if len(self.buffer) >= self.max_size:
            del self.buffer[0]

        self.buffer.append(item)

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))


def train():
    max_buffer_size = int(1e5)
    batch_size = 32
    max_grad_norm = 1.0
    num_of_iterations = int(5e6)
    evaluation_episodes = 20
    start_epsilon = 0.9
    min_epsilon = 0.1
    # decay to 0.1 over the first 500k iterations
    epsilon_decay_rate = (start_epsilon - min_epsilon) / 5e5
    learning_rate = 5.0e-4
    target_network_update_freq = 5000
    stacked_frame_count = 4
    network_save_freq = 20000  # every 20000 iterations
    recording_frequency = 1 / 100

    model1 = ZombieFightModel()
    model2 = ZombieFightModel()

    parser = argparse.ArgumentParser()
    parser.add_argument("--init-model1")
    parser.add_argument("--init-model2")
    parser.add_argument("--initial-iteration", type=int, default=0)
    args = parser.parse_args()

    if args.init_model1:
        assert args.init_model2

        print("Loading initial model weights from files")
        model1.load_state_dict(torch.load(args.init_model1, map_location="cpu"))
        model2.load_state_dict(torch.load(args.init_model2, map_location="cpu"))

    model1_target = ZombieFightModel()
    model1_target.load_state_dict(model1.state_dict())
    model2_target = ZombieFightModel()
    model2_target.load_state_dict(model2.state_dict())

    print(
        f"Trainable parameters: {sum([p.numel() for p in model1.parameters() if p.requires_grad])}"
    )
    model1.train()
    model2.train()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model1.to(device)
    model2.to(device)
    model1_target.to(device)
    model2_target.to(device)

    memory = ReplayMemory(max_buffer_size)

    loss_fn = torch.nn.MSELoss()
    mae_fn = torch.nn.L1Loss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)

    writer = SummaryWriter()

    os.makedirs("models", exist_ok=True)
    os.makedirs("videos", exist_ok=True)

    @torch.no_grad()
    def pick_greedily(state):
        images, others = state

        images = images.to(device).unsqueeze(0)
        others = others.to(device).unsqueeze(0)

        action_values1 = model1(images=images, others=others).squeeze(0)
        action_values2 = model2(images=images, others=others).squeeze(0)

        action_values = (action_values1 + action_values2) / 2

        return torch.argmax(action_values, -1).item()

    class Evaluator:
        def __init__(self):
            self.episode_rewards = []

        def evaluate_until_episodes_done(self, client):
            while len(self.episode_rewards) < evaluation_episodes:
                episode_reward = 0
                done = False
                state_constructor = StackedStateConstructor(stacked_frame_count)
                state_constructor.append(create_observation(client.reset()))

                while not done:
                    action = pick_greedily(state_constructor.current())
                    next_state_raw, reward, done = client.step(ACTIONS[action])

                    episode_reward += reward
                    state_constructor.append(create_observation(next_state_raw))

                self.episode_rewards.append(episode_reward)

        def evaluate(self):
            print("Starting evaluation")
            threads = [Thread(target=self.evaluate_until_episodes_done, args=(client,)) for client in clients]

            [thread.start() for thread in threads]
            [thread.join() for thread in threads]

            print("Evaluation finished")

            return self.episode_rewards

    class Trainer:
        def __init__(self, initial_iteration):
            self.n_iter = initial_iteration
            self.best_model_reward = -1e9
            self.epsilon = start_epsilon
            self.train_step_moving_average = MovingAverageCalculator(2000)
            self.data_collection_stopped = False
            self.data_collection_threads = []
            self.episode_counter = itertools.count(1)
            self.collect_data_moving_average = MovingAverageCalculator(2000)
            self.episode_reward_moving_average = MovingAverageCalculator(30)
            self.recorded_episode_counter = itertools.count(1)

        def train_on_batch(self):
            if random.random() < 0.5:
                optimizer = optimizer1
                model = model1
            else:
                optimizer = optimizer2
                model = model2

            optimizer.zero_grad()

            batch = memory.sample_batch(batch_size)

            images = torch.stack([timestep[0][0] for timestep in batch], dim=0)
            others = torch.stack([timestep[0][1] for timestep in batch], dim=0)

            action = torch.tensor([timestep[1] for timestep in batch], dtype=torch.int64, device=device)
            next_images = torch.stack([timestep[2][0] for timestep in batch], dim=0)
            next_others = torch.stack([timestep[2][1] for timestep in batch], dim=0)

            reward = torch.tensor(
                [timestep[3] for timestep in batch], dtype=torch.float32, device=device
            )
            done = torch.tensor([timestep[4] for timestep in batch], dtype=torch.int32, device=device)

            images = images.to(device)
            others = others.to(device)
            next_images = next_images.to(device)
            next_others = next_others.to(device)

            assert images.shape == (len(batch), stacked_frame_count, 84, 84)
            assert others.shape == (len(batch), 7)
            assert next_images.shape == (len(batch), stacked_frame_count, 84, 84)
            assert next_others.shape == (len(batch), 7)
            assert action.shape == (len(batch),)
            assert reward.shape == (len(batch),)
            assert done.shape == (len(batch),)

            with torch.no_grad():
                max_index = torch.argmax(model1_target(images=next_images, others=next_others), dim=-1)
                next_state_estimate = torch.gather(
                    model2_target(images=next_images, others=next_others), -1, max_index.unsqueeze(1)
                ).squeeze(1)
                assert next_state_estimate.shape == (len(batch),)
                next_state_estimate.detach()

            target = reward + next_state_estimate * (1 - done)
            current_estimate = torch.gather(model(images=images, others=others), -1, action.unsqueeze(1)).squeeze(1)

            loss = loss_fn(current_estimate, target)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            writer.add_scalar("Loss/train", loss.item(), self.n_iter)
            writer.add_scalar("Unclipped gradient norm", grad_norm.item(), self.n_iter)

            with torch.no_grad():
                mae = mae_fn(current_estimate, target)
                writer.add_scalar("Mean absolute error", mae.item(), self.n_iter)

        def train(self):
            self.start_data_collection_threads()

            while self.n_iter < num_of_iterations:
                if len(memory) < batch_size:
                    continue

                t1 = time.time()
                self.train_on_batch()
                t2 = time.time()

                self.n_iter += 1
                self.train_step_moving_average.insert_value((t2 - t1) * 1000)

                if self.n_iter % 1000 == 0:
                    print(f"Average time per train step: {self.train_step_moving_average.get():.3f} ms")
                    print(f"Average time per data collection step: {self.collect_data_moving_average.get():.3f} ms")

                self.epsilon = max(min_epsilon, self.epsilon - epsilon_decay_rate)
                writer.add_scalar("Epsilon", self.epsilon, self.n_iter)

                if self.n_iter % target_network_update_freq == 0:
                    print("Updating target networks")
                    model1_target.load_state_dict(model1.state_dict())
                    model2_target.load_state_dict(model2.state_dict())

                if self.n_iter % network_save_freq == 0:
                    self.stop_data_collection_threads()

                    episode_rewards = Evaluator().evaluate()
                    average_reward = sum(episode_rewards) / len(episode_rewards)
                    print(f"Average reward per episode after {self.n_iter} iterations: {average_reward}")
                    writer.add_scalar("Evaluation/Average reward", average_reward, self.n_iter)

                    print("Saving weights")
                    torch.save(model1.state_dict(), f"models/zf-model1-{self.n_iter}.pt")
                    torch.save(model2.state_dict(), f"models/zf-model2-{self.n_iter}.pt")

                    if average_reward > self.best_model_reward:
                        print("Current model is better than the previous best. Saving it as best model.")
                        torch.save(model1.state_dict(), "models/zf-model1-best.pt")
                        torch.save(model2.state_dict(), "models/zf-model2-best.pt")
                        self.best_model_reward = average_reward

                    self.start_data_collection_threads()

            self.stop_data_collection_threads()
            print("Training finished. Starting evaluation.")

            episode_rewards = Evaluator().evaluate()
            print(f"Average reward per episode: {sum(episode_rewards) / len(episode_rewards)}")
            plt.hist(episode_rewards, 50)
            plt.savefig("reward_distribution.png")

            print("Saving final model weights")
            torch.save(model1.state_dict(), "models/zf-model1.pt")
            torch.save(model2.state_dict(), "models/zf-model2.pt")

        def collect_data_until_stopped(self, client):
            while not self.data_collection_stopped:
                record_episode = random.random() < recording_frequency
                if record_episode:
                    recorder = EpisodeRecorder(next(self.recorded_episode_counter))

                episode_reward = 0
                episode_start_time = time.time()
                done = False
                step_count = 0

                state_constructor = StackedStateConstructor(stacked_frame_count)
                initial_state = create_observation(client.reset())
                state_constructor.append(initial_state)

                if record_episode:
                    recorder.record_state(initial_state.image)

                while not done:
                    t1 = time.time()

                    step_count += 1

                    stacked_state = state_constructor.current()

                    action = self.epsilon_greedy(stacked_state)

                    next_state_raw, reward, done = client.step(ACTIONS[action])

                    episode_reward += reward
                    next_state = create_observation(next_state_raw)
                    state_constructor.append(next_state)
                    memory.append((stacked_state, action, state_constructor.current(), reward, done))

                    if record_episode:
                        recorder.record_state(next_state.image)

                    t2 = time.time()
                    self.collect_data_moving_average.insert_value((t2 - t1) * 1000)

                episode_order = next(self.episode_counter)
                if episode_order % 25 == 0:
                    print(f"{episode_order} episodes finished")

                episode_duration = time.time() - episode_start_time
                print(f"Episode finished with reward {episode_reward}, took {episode_duration:.1f} seconds, "
                      f"TPS {step_count / episode_duration:.1f}")

                self.episode_reward_moving_average.insert_value(episode_reward)
                writer.add_scalar("Episode reward moving average", self.episode_reward_moving_average.get(),
                                  episode_order)

                if record_episode:
                    recorder.finish()
                    print(f"Finished recording episode with {step_count} steps")

        def start_data_collection_threads(self):
            self.data_collection_stopped = False
            self.data_collection_threads = [
                Thread(target=self.collect_data_until_stopped, args=(client,)) for client in clients
            ]

            [thread.start() for thread in self.data_collection_threads]

        def stop_data_collection_threads(self):
            self.data_collection_stopped = True
            [thread.join() for thread in self.data_collection_threads]

        def epsilon_greedy(self, state):
            if random.random() < self.epsilon:
                return pick_greedily(state)
            else:
                return random.randint(0, len(ACTIONS) - 1)

    Trainer(args.initial_iteration).train()

    writer.close()


if __name__ == "__main__":
    train()
