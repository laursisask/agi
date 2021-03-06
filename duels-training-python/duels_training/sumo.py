import argparse
import copy
import csv
import datetime
import json
import logging
import math
import os
import random
import statistics
from threading import Thread

import cv2
import torch
from terminator import TerminatorSumo, Action
from torch.utils.tensorboard import SummaryWriter

from duels_training.incremental_stats_calculator import IncrementalStatsCalculator
from duels_training.logger import configure_logger
from duels_training.opponent_sampler import OpponentSampler
from duels_training.ppo import collect_data_and_train
from duels_training.stats_utils import normal_log_probs, inverse_log_prob, categorical_entropy, bernoulli_entropy, \
    normal_entropy
from duels_training.sumo_maps import MAPS
from duels_training.sumo_model import SumoModel
from duels_training.sumo_policy import compute_log_prob_dists, sample_action
from duels_training.preprocessing import transform_raw_state


class EpisodeStatsAggregator:
    def __init__(self, run_id, tensorboard):
        self.tensorboard = tensorboard

        path = f"artifacts/{run_id}/metrics.csv"
        if os.path.exists(path):
            self.metrics_file = open(path, "a")
            self.csv_writer = csv.writer(self.metrics_file)
        else:
            self.metrics_file = open(path, "w")
            self.csv_writer = csv.writer(self.metrics_file)
            self.csv_writer.writerow(["iteration", "reward", "length", "win_rate", "hits_done", "hits_received",
                                      "attack_step_fraction", "jump_step_fraction", "sprint_step_fraction"])

    def __call__(self, global_it, trajectories):
        avg_reward = statistics.mean([sum(trajectory.rewards) for trajectory in trajectories])
        logging.info(f"Average reward was {avg_reward:.3f}")
        self.tensorboard.add_scalar("Average reward", avg_reward, global_it)

        avg_length = statistics.mean([len(trajectory.actions) for trajectory in trajectories])
        logging.info(f"Average episode length was {avg_length:.1f} steps")
        self.tensorboard.add_scalar("Average length", avg_length, global_it)

        avg_win_rate = self.compute_average_metadata_value(trajectories, "win")
        logging.info(f"Average win rate per episode was {avg_win_rate:.2f}")
        self.tensorboard.add_scalar("Average win rate", avg_win_rate, global_it)

        avg_hits_done = self.compute_average_metadata_value(trajectories, "hits_done")
        logging.info(f"Average number of hits done per episode was {avg_hits_done:.2f}")
        self.tensorboard.add_scalar("Average number of hits done", avg_hits_done, global_it)

        avg_hits_received = self.compute_average_metadata_value(trajectories, "hits_received")
        logging.info(f"Average number of hits received per episode was {avg_hits_received:.2f}")
        self.tensorboard.add_scalar("Average number of hits received", avg_hits_received, global_it)

        avg_attack_step_fraction = self.compute_fraction_of_steps(
            trajectories,
            lambda action: tensor_to_action(action).attacking
        )
        logging.info(f"Average attack step fraction per episode was {avg_attack_step_fraction:.2f}")
        self.tensorboard.add_scalar("Average attack step fraction", avg_attack_step_fraction, global_it)

        avg_jump_step_fraction = self.compute_fraction_of_steps(
            trajectories,
            lambda action: tensor_to_action(action).jumping
        )
        logging.info(f"Average jumping step fraction per episode was {avg_jump_step_fraction:.2f}")
        self.tensorboard.add_scalar("Average jumping step fraction", avg_jump_step_fraction, global_it)

        avg_sprint_step_fraction = self.compute_fraction_of_steps(
            trajectories,
            lambda action: tensor_to_action(action).sprinting
        )
        logging.info(f"Average sprinting step fraction per episode was {avg_sprint_step_fraction:.2f}")
        self.tensorboard.add_scalar("Average sprinting step fraction", avg_sprint_step_fraction, global_it)

        self.csv_writer.writerow([global_it, avg_reward, avg_length, avg_win_rate, avg_hits_done, avg_hits_received,
                                  avg_attack_step_fraction, avg_jump_step_fraction, avg_sprint_step_fraction])
        self.metrics_file.flush()

    @staticmethod
    def compute_average_metadata_value(trajectories, key):
        values = []

        for trajectory in trajectories:
            episode_cumulative = 0
            for metadata in trajectory.metadatas:
                if key in metadata:
                    episode_cumulative += metadata[key]
            values.append(episode_cumulative)

        return statistics.mean(values)

    @staticmethod
    def compute_fraction_of_steps(trajectories, filter_fn):
        values = []

        for trajectory in trajectories:
            matching_steps = 0
            for action in trajectory.actions:
                if filter_fn(action):
                    matching_steps += 1
            values.append(matching_steps / len(trajectory))

        return statistics.mean(values)


def get_available_maps(global_iteration, add_maps_start=200, add_maps_end=600):
    if global_iteration < add_maps_start:
        available_maps = MAPS[:1]
    elif global_iteration > add_maps_end:
        available_maps = MAPS
    else:
        max_index = math.floor(
            (global_iteration - add_maps_start) / (add_maps_end - add_maps_start) * (len(MAPS) - 1) + 1
        )

        available_maps = MAPS[:max_index]

    return available_maps


class SumoEnv:
    def __init__(self, run_id, device, client1, client2, opponent_sampler, get_global_iteration,
                 get_last_num_of_episodes):
        self.run_id = run_id
        self.device = device
        self.client1 = client1
        self.client2 = client2
        self.get_global_iteration = get_global_iteration
        self.opponent_sampler = opponent_sampler
        self.get_last_num_of_episodes = get_last_num_of_episodes
        self.record_episode = False
        self.recorder = None
        self.opponent_thread = None
        self.current_map = None
        self.map_episode_counter = 0

    def reset(self):
        if self.record_episode and self.recorder.isOpened():
            self.recorder.release()

        self.record_episode = random.random() < 0.002

        available_maps = get_available_maps(self.get_global_iteration())
        episodes_per_map = self.get_last_num_of_episodes() / len(available_maps)
        if self.current_map is None or self.map_episode_counter >= episodes_per_map:
            self.current_map = random.choice(available_maps)
            self.map_episode_counter = 0

        self.map_episode_counter += 1

        session = self.client1.create_session()

        opponent_model = self.opponent_sampler.sample()

        if self.opponent_thread:
            self.opponent_thread.join()

        self.opponent_thread = Thread(target=self.play_as_opponent, args=(opponent_model, session), daemon=True)
        self.opponent_thread.start()

        raw_observation = self.client1.reset(
            session=session,
            randomization_factor=1.0,
            map_name=self.current_map,
            random_teleport=self.random_teleport_active()
        )

        if self.record_episode:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            file = f"artifacts/{self.run_id}/videos/{datetime.datetime.now().isoformat()}.mp4"
            os.makedirs(os.path.dirname(file), exist_ok=True)
            self.recorder = cv2.VideoWriter(file, fourcc, 30, (84, 84), False)
            self.recorder.write(cv2.flip(raw_observation.camera.numpy(), 0))

        return transform_raw_state(raw_observation)

    def step(self, action):
        terminator_action = tensor_to_action(action)
        raw_observation, game_reward, done, metadata = self.client1.step(terminator_action)

        if self.record_episode:
            self.recorder.write(cv2.flip(raw_observation.camera.numpy(), 0))

            if done:
                self.recorder.release()

        total_reward = game_reward + self.calculate_exploration_reward(metadata, terminator_action)

        return transform_raw_state(raw_observation), total_reward, done, metadata

    def calculate_exploration_reward(self, metadata, action):
        return self.calculate_hit_reward(metadata) + self.calculate_attack_reward(action)

    def calculate_hit_reward(self, metadata):
        if "hits_done" not in metadata and "hits_received" not in metadata:
            return 0

        global_it = self.get_global_iteration()

        annealing_start = 0
        annealing_end = 3000

        if global_it > annealing_end:
            return 0

        if global_it < annealing_start:
            exploration_coefficient = 1
        else:
            exploration_coefficient = (annealing_end - global_it) / (annealing_end - annealing_start)

        return exploration_coefficient * 10 * (metadata.get("hits_done", 0) - metadata.get("hits_received", 0))

    def calculate_attack_reward(self, action):
        if not action.attacking:
            return 0

        increase_start = 1000
        increase_end = 2000
        decrease_end = 3000

        if increase_start < self.get_global_iteration() < increase_end:
            return -0.05 * (increase_end - self.get_global_iteration()) / (increase_end - increase_start)
        elif increase_end <= self.get_global_iteration() < decrease_end:
            return -0.05 * (decrease_end - self.get_global_iteration()) / (decrease_end - increase_end)
        else:
            return 0

    def play_as_opponent(self, model, session):
        observation = transform_raw_state(self.client2.reset(
            session=session,
            randomization_factor=1.0,
            map_name=self.current_map,
            random_teleport=self.random_teleport_active()
        ))

        policy_state = PolicyState(
            model=model,
            device=self.device,
            observation=observation
        )

        done = False
        while not done:
            action = policy_state.sample_action()
            next_observation, _, done, _ = self.client2.step(tensor_to_action(action))
            policy_state.update(transform_raw_state(next_observation))

    def random_teleport_active(self):
        return 1000 < self.get_global_iteration() < 3000


@torch.jit.script
def compute_entropies(model_output: torch.Tensor):
    (forward_dist, left_dist, jumping_dist, attacking_dist, sprinting_dist,
     yaw_mean, yaw_std, pitch_mean, pitch_std) = compute_log_prob_dists(model_output)

    batch_size: int = model_output.size(0)

    forward_entropy = categorical_entropy(forward_dist)
    assert forward_entropy.shape == (batch_size,)

    left_entropy = categorical_entropy(left_dist)
    assert left_entropy.shape == (batch_size,)

    jumping_entropy = bernoulli_entropy(jumping_dist)
    assert jumping_entropy.shape == (batch_size,)

    attacking_entropy = bernoulli_entropy(attacking_dist)
    assert attacking_entropy.shape == (batch_size,)

    sprinting_entropy = bernoulli_entropy(sprinting_dist)
    assert sprinting_entropy.shape == (batch_size,)

    yaw_entropy = normal_entropy(yaw_std)
    assert yaw_entropy.shape == (batch_size,)

    pitch_entropy = normal_entropy(pitch_std)
    assert pitch_entropy.shape == (batch_size,)

    return (forward_entropy + left_entropy + jumping_entropy + attacking_entropy +
            sprinting_entropy + yaw_entropy + pitch_entropy) / 7


def action_log_probs(model_output: torch.Tensor, actions: Action):
    batch_size = actions.forward.size(0)
    assert actions.forward.shape == (batch_size,)
    assert actions.left.shape == (batch_size,)
    assert actions.jumping.shape == (batch_size,)
    assert actions.attacking.shape == (batch_size,)
    assert actions.sprinting.shape == (batch_size,)
    assert actions.delta_yaw.shape == (batch_size,)
    assert actions.delta_pitch.shape == (batch_size,)

    (forward_dist, left_dist, jumping_dist, attacking_dist, sprinting_dist,
     yaw_mean, yaw_std, pitch_mean, pitch_std) = compute_log_prob_dists(model_output)

    forward_index = actions.forward.unsqueeze(1).to(torch.int64) + 1
    forward_log_probs = forward_dist.gather(dim=1, index=forward_index).squeeze(1)
    assert forward_log_probs.shape == (batch_size,)

    left_index = actions.left.unsqueeze(1).to(torch.int64) + 1
    left_log_probs = left_dist.gather(dim=1, index=left_index).squeeze(1)
    assert left_log_probs.shape == (batch_size,)

    jumping_log_probs = torch.where(actions.jumping, jumping_dist, inverse_log_prob(jumping_dist))
    assert jumping_log_probs.shape == (batch_size,)

    attacking_log_probs = torch.where(actions.attacking, attacking_dist, inverse_log_prob(attacking_dist))
    assert attacking_log_probs.shape == (batch_size,)

    sprinting_log_probs = torch.where(actions.sprinting, sprinting_dist, inverse_log_prob(sprinting_dist))
    assert sprinting_log_probs.shape == (batch_size,)

    yaw_log_probs = normal_log_probs(yaw_mean, yaw_std, actions.delta_yaw)
    pitch_log_probs = normal_log_probs(pitch_mean, pitch_std, actions.delta_pitch)

    return (forward_log_probs + left_log_probs + jumping_log_probs + attacking_log_probs +
            sprinting_log_probs + yaw_log_probs + pitch_log_probs)


def action_to_tensor(action: Action):
    return torch.tensor([action.forward, action.left, action.jumping, action.attacking,
                         action.sprinting, action.delta_yaw, action.delta_pitch], dtype=torch.float32)


def tensor_to_action(tensor) -> Action:
    if tensor.shape == (7,):
        forward, left, jumping, attacking, sprinting, delta_yaw, delta_pitch = tensor

        return Action(
            forward=int(forward.item()),
            left=int(left.item()),
            jumping=bool(jumping.item()),
            attacking=bool(attacking.item()),
            sprinting=bool(sprinting.item()),
            delta_yaw=delta_yaw.item(),
            delta_pitch=delta_pitch.item()
        )
    elif tensor.size(-1) == 7:
        forward, left, jumping, attacking, sprinting, delta_yaw, delta_pitch = tensor.split(split_size=1, dim=-1)

        return Action(
            forward=forward.to(torch.int32).squeeze(-1),
            left=left.to(torch.int32).squeeze(-1),
            jumping=jumping.to(torch.bool).squeeze(-1),
            attacking=attacking.to(torch.bool).squeeze(-1),
            sprinting=sprinting.to(torch.bool).squeeze(-1),
            delta_yaw=delta_yaw.squeeze(-1),
            delta_pitch=delta_pitch.squeeze(-1)
        )
    else:
        raise ValueError("Given tensor of invalid shape")


class PolicyState:
    def __init__(self, model, device, observation):
        self.model = model
        self.device = device
        self.last_state = None
        self.policy_output = None

        self.update(observation)

    @torch.no_grad()
    def update(self, observation):
        policy_output, _, self.last_state = self.model(
            observation.unsqueeze(0).unsqueeze(0).to(self.device),
            initial_state=self.last_state,
            compute_value_estimates=False
        )

        self.policy_output = policy_output.to(torch.device("cpu")).squeeze(1)

    @torch.no_grad()
    def sample_action(self):
        return action_to_tensor(sample_action(self.policy_output))


class Policy:
    def __init__(self, model, device):
        super().__init__()

        self.model = model
        self.device = device

    def action_log_prob(self, policy_dists, actions):
        return action_log_probs(model_output=policy_dists, actions=tensor_to_action(actions))

    def get_initial_state(self, observation):
        return PolicyState(self.model, self.device, observation)

    def entropy(self, policy_dists):
        return compute_entropies(policy_dists)


def train(initial_model, initial_optimizer, initial_reward_stats, start_global_iteration, start_train_iteration,
          log_dir, run_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using {device} device")

    os.makedirs(f"artifacts/{run_id}", exist_ok=True)

    if log_dir is None:
        metrics = SummaryWriter()
        log_dir = metrics.log_dir
    else:
        metrics = SummaryWriter(log_dir=log_dir)

    model = SumoModel()
    if initial_model is not None:
        logging.info(f"Loading model weights from {initial_model}")
        model.load_state_dict(torch.load(initial_model, map_location=torch.device("cpu")))

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    if initial_optimizer is not None:
        logging.info(f"Loading optimizer weights from {initial_optimizer}")
        state_dict = torch.load(initial_optimizer)
        optimizer.load_state_dict(state_dict)

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device).contiguous()

    model.to(device)

    reward_stats = IncrementalStatsCalculator()
    if initial_reward_stats is not None:
        logging.info(f"Loading reward stats from {initial_reward_stats}")
        with open(initial_reward_stats, "r") as f:
            reward_stats.load(f)

    global_iteration = 1 if start_global_iteration is None else start_global_iteration

    def post_iteration_callback(new_global_iteration, train_iteration):
        model_copy = copy.deepcopy(model)
        model_copy.to(torch.device("cpu"))
        model_copy.eval()

        os.makedirs(f"artifacts/{run_id}/models", exist_ok=True)
        torch.save(model_copy.state_dict(), f"artifacts/{run_id}/models/{new_global_iteration}.pt")

        with open(f"artifacts/{run_id}/last_iteration.json", "w") as file:
            last_iteration = {
                "global_iteration": new_global_iteration,
                "train_iteration": train_iteration,
                "log_dir": log_dir
            }
            json.dump(last_iteration, file)

        with open(f"artifacts/{run_id}/reward_stats.json", "w") as f:
            reward_stats.dump(f)

        nonlocal global_iteration
        global_iteration = new_global_iteration

    episode_stats_aggregator = EpisodeStatsAggregator(run_id, metrics)

    def load_model(index):
        new_model = SumoModel()
        new_model.load_state_dict(torch.load(f"artifacts/{run_id}/models/{index}.pt", map_location=torch.device("cpu")))
        new_model.to(device)
        return new_model

    opponent_sampler = OpponentSampler(
        last_model=model,
        get_global_iteration=lambda: global_iteration,
        load_model=load_model,
        opponent_sampling_index=0.5
    )

    last_num_of_episodes = 50

    def post_data_collect_callback(global_it, trajectories):
        episode_stats_aggregator(global_it, trajectories)

        nonlocal last_num_of_episodes
        last_num_of_episodes = len(trajectories)

    num_clients = 6
    assert num_clients % 2 == 0

    clients = []
    for port in range(6660, 6660 + num_clients):
        logging.info(f"Connecting to terminator on localhost:{port}")
        client = TerminatorSumo()
        client.connect(("localhost", port))
        clients.append(client)

    envs = [SumoEnv(
        run_id=run_id,
        device=device,
        client1=clients[i],
        client2=clients[i + 1],
        opponent_sampler=opponent_sampler,
        get_global_iteration=lambda: global_iteration,
        get_last_num_of_episodes=lambda: last_num_of_episodes
    ) for i in range(0, num_clients, 2)]

    collect_data_and_train(
        device=device,
        policy=Policy(model, device),
        envs=envs,
        model=model,
        optimizer=optimizer,
        metrics=metrics,
        reward_stats=reward_stats,
        iterations=50000,
        steps_per_iteration=65536,
        bootstrap_length=5,
        discount_factor=0.99,
        gae_param=0.95,
        num_epochs=4,
        batch_size=1280,
        parallel_sequences=64,
        backpropagation_steps=20,
        clip_range=0.1,
        max_grad_norm=1.0,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        post_iteration_callback=post_iteration_callback,
        post_data_collect_callback=post_data_collect_callback,
        start_global_iteration=1 if start_global_iteration is None else start_global_iteration,
        start_train_iteration=0 if start_train_iteration is None else start_train_iteration,
        assets_dir=f"artifacts/{run_id}"
    )


def main():
    configure_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)

    args = parser.parse_args()

    if os.path.exists(f"artifacts/{args.id}/last_iteration.json"):
        logging.info("Continuing training")

        initial_model = f"artifacts/{args.id}/last_model.pt"
        initial_optimizer = f"artifacts/{args.id}/last_optimizer.pt"
        initial_reward_stats = f"artifacts/{args.id}/reward_stats.json"

        with open(f"artifacts/{args.id}/last_iteration.json", "r") as file:
            last_iteration = json.load(file)

        start_global_iteration = last_iteration["global_iteration"] + 1
        start_train_iteration = last_iteration["train_iteration"]
        log_dir = last_iteration["log_dir"]
    else:
        logging.info("Starting training from scratch")

        initial_model = None
        initial_optimizer = None
        initial_reward_stats = None
        start_global_iteration = None
        start_train_iteration = None
        log_dir = None

    train(
        initial_model,
        initial_optimizer,
        initial_reward_stats,
        start_global_iteration,
        start_train_iteration,
        log_dir,
        args.id
    )


if __name__ == "__main__":
    main()
