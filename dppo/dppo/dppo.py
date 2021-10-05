import copy
import logging
import math
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from multiprocessing import Pool
from threading import Thread
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import ReduceOp


def configure_logger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    root.handlers.clear()
    handler = logging.StreamHandler(stream=sys.stdout)

    formatter = logging.Formatter(
        fmt=f"[{dist.get_rank()}] %(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    root.addHandler(handler)


@dataclass
class Trajectory:
    observations: List
    actions: List
    rewards: List

    def __len__(self):
        return len(self.observations)

    def __post_init__(self):
        assert len(self.observations) == len(self.actions) == len(self.rewards)


class DataCollector:
    def __init__(self, policy, env):
        self.policy = policy
        self.env = env

    def run_episode(self):
        start_time = time.time()

        observation = self.env.reset()
        policy_state = self.policy.get_initial_state(observation)
        done = False
        cumulative_reward = 0
        steps_done = 0

        observations = []
        actions = []
        rewards = []

        while not done:
            action = policy_state.sample_action()

            next_observation, reward, done, _ = self.env.step(action)
            cumulative_reward += reward

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)

            policy_state.update(next_observation)
            observation = next_observation

            steps_done += 1

        end_time = time.time()
        seconds_elapsed = end_time - start_time

        logging.info(f"Episode ended with cumulative reward of {cumulative_reward}, "
                     f"took {seconds_elapsed:.2f} seconds, "
                     f"TPS was {steps_done / seconds_elapsed:.2f}")

        return Trajectory(observations=observations, actions=actions, rewards=rewards)


def compute_value_estimates(device, critic, observations, steps_per_call) -> torch.Tensor:
    cpu = torch.device("cpu")

    batch_value_estimates = []
    last_state = None

    for batch in observations.split(split_size=steps_per_call, dim=1):
        value_estimates, last_state = critic(batch.to(device), initial_state=last_state)
        batch_value_estimates.append(value_estimates.to(cpu))

    value_estimates = torch.cat(batch_value_estimates, dim=1)

    assert value_estimates.size(0) == observations.size(0)
    assert value_estimates.size(1) == observations.size(1)

    return value_estimates


def compute_n_step_returns(value_estimates: torch.Tensor, rewards: List[List[float]], n: int) -> torch.Tensor:
    returns = torch.zeros(value_estimates.shape)

    num_of_trajectories = len(rewards)
    for i in range(num_of_trajectories):
        trajectory_length = len(rewards[i])
        last_n_rewards = 0

        for t in reversed(range(trajectory_length)):
            last_n_rewards += rewards[i][t]

            if t + n < trajectory_length:
                last_n_rewards -= rewards[i][t + n]

            if t + n < trajectory_length:
                ret = last_n_rewards + value_estimates[i][t + n].item()
            else:
                ret = last_n_rewards

            returns[i][t] = ret

    return returns


@torch.no_grad()
def create_dataset(device, critic, trajectories, bootstrap_length, batch_size=64, steps_per_call=32):
    trajectories = sorted(trajectories, key=len)

    def collate_fn(batch):
        observations = pad_sequence(
            [torch.stack(trajectory.observations) for trajectory in batch],
            batch_first=True
        )

        actions = [trajectory.actions for trajectory in batch]
        rewards = [trajectory.rewards for trajectory in batch]

        return observations, actions, rewards

    data_loader = DataLoader(trajectories, batch_size=batch_size, collate_fn=collate_fn)

    dataset = []
    for observations, actions, rewards in data_loader:
        value_estimates = compute_value_estimates(
            device=device,
            critic=critic,
            observations=observations,
            steps_per_call=steps_per_call
        )

        returns = compute_n_step_returns(value_estimates, rewards, bootstrap_length)

        advantages = returns - value_estimates

        for t_observations, t_actions, t_rewards, t_returns, t_advantages in zip(
                observations,
                actions,
                rewards,
                returns,
                advantages
        ):
            length = len(t_actions)

            dataset.append((
                t_observations[:length],
                torch.stack(t_actions),
                t_returns[:length],
                t_advantages[:length]
            ))

    return dataset


def standardize(x, mask):
    n = torch.sum(mask)
    assert n > 1

    mean = torch.sum(x * mask) / n
    std = torch.sum(((x - mean) * mask) ** 2) / n + 1e-7

    return (x - mean) / std


def is_primary():
    return dist.get_rank() == 0


def average_grads(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.reduce(param.data, 0, op=ReduceOp.SUM)
        param.data /= size


def sync_params(model):
    for param in model.parameters():
        dist.broadcast(param.data, 0)


def train_on_batch(device, optimizer, model, model_old, policy, batch, backpropagation_steps,
                   clip_range, max_grad_norm, value_coefficient, entropy_coefficient):
    model.zero_grad()

    all_observations, all_actions, all_returns, all_advantages, all_mask = batch

    all_advantages = standardize(all_advantages, all_mask)

    backward_steps = math.ceil(all_observations.size(1) / backpropagation_steps)

    actor_loss_sum = 0
    critic_loss_sum = 0
    entropy_loss_sum = 0
    loss_sum = 0

    last_state = None
    last_state_old = None

    for observations, actions, returns, advantages, mask in zip(
            all_observations.split(backpropagation_steps, dim=1),
            all_actions.split(backpropagation_steps, dim=1),
            all_returns.split(backpropagation_steps, dim=1),
            all_advantages.split(backpropagation_steps, dim=1),
            all_mask.split(backpropagation_steps, dim=1)
    ):
        policy_dists, value_estimates, last_state = model(observations, initial_state=last_state)

        log_probs = policy.action_log_prob(policy_dists, actions)
        assert log_probs.shape == advantages.shape

        with torch.no_grad():
            policy_dists_old, last_state_old = model_old(
                observations,
                initial_state=last_state_old,
                compute_value_estimates=False
            )

        log_probs_old = policy.action_log_prob(policy_dists_old, actions)
        assert log_probs_old.shape == advantages.shape

        ratios = torch.exp(log_probs - log_probs_old)
        num_of_samples = torch.sum(mask)
        assert num_of_samples >= 1

        surrogate_objective = torch.sum(torch.min(
            ratios * advantages,
            torch.clamp(ratios, 1 - clip_range, 1 + clip_range) * advantages
        ) * mask) / num_of_samples

        actor_loss = -surrogate_objective
        assert actor_loss.numel() == 1
        assert not actor_loss.isnan()

        critic_loss = torch.sum(F.mse_loss(value_estimates, returns, reduction="none") * mask) / num_of_samples
        assert not critic_loss.isnan()
        assert critic_loss.numel() == 1

        entropies = policy.entropy(policy_dists)
        assert entropies.shape == mask.shape

        entropy = torch.sum(entropies * mask) / num_of_samples
        entropy_loss = -entropy
        assert not entropy_loss.isnan()
        assert entropy_loss.numel() == 1

        loss = actor_loss + value_coefficient * critic_loss + entropy_coefficient * entropy_loss
        (loss / backward_steps).backward()

        actor_loss_sum += actor_loss.item()
        critic_loss_sum += critic_loss.item()
        entropy_loss_sum += entropy_loss.item()
        loss_sum += loss.item()

        last_state.detach_()

    average_grads(model)

    if is_primary():
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

    sync_params(model)

    worker_loss = torch.tensor([loss_sum / backward_steps], dtype=torch.float32, device=device)
    dist.reduce(worker_loss, 0, op=ReduceOp.SUM)
    avg_loss = worker_loss.item() / dist.get_world_size()

    worker_actor_loss = torch.tensor([actor_loss_sum / backward_steps], dtype=torch.float32, device=device)
    dist.reduce(worker_actor_loss, 0, op=ReduceOp.SUM)
    avg_actor_loss = worker_actor_loss.item() / dist.get_world_size()

    worker_critic_loss = torch.tensor([critic_loss_sum / backward_steps], dtype=torch.float32, device=device)
    dist.reduce(worker_critic_loss, 0, op=ReduceOp.SUM)
    avg_critic_loss = worker_critic_loss.item() / dist.get_world_size()

    worker_entropy_loss = torch.tensor([entropy_loss_sum / backward_steps], dtype=torch.float32, device=device)
    dist.reduce(worker_entropy_loss, 0, op=ReduceOp.SUM)
    avg_entropy_loss = worker_entropy_loss.item() / dist.get_world_size()

    if is_primary():
        return (
            avg_loss,
            avg_actor_loss,
            avg_critic_loss,
            avg_entropy_loss,
            grad_norm.item()
        )
    else:
        return 0, 0, 0, 0, 0


def train(device, dataset, model, policy, optimizer, metrics, start_iteration, num_epochs, batch_size,
          backpropagation_steps, clip_range, max_grad_norm, value_coefficient, entropy_coefficient):
    if is_primary():
        logging.info(f"Training on {len(dataset)} trajectories")

    model_old = copy.deepcopy(model)

    def collate_fn(batch):
        # batch: list of tuples where each tuple is
        # (observations: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor, advantages: torch.Tensor)
        observations, actions, returns, advantages = zip(*batch)
        assert len(observations) == len(actions) == len(advantages) == batch_size

        lengths = torch.tensor([len(trajectory_observations) for trajectory_observations in observations],
                               dtype=torch.int64)
        max_length = torch.max(lengths).item()

        mask = torch.arange(max_length).reshape([1, max_length]).expand(batch_size, -1) < lengths.unsqueeze(-1)

        observations = pad_sequence(observations, batch_first=True)
        actions = pad_sequence(actions, batch_first=True)
        returns = pad_sequence(returns, batch_first=True)
        advantages = pad_sequence(advantages, batch_first=True)

        observations = observations.to(device)
        actions = actions.to(device)
        returns = returns.to(device)
        advantages = advantages.to(device)
        mask = mask.to(device)

        assert observations.size(0) == actions.size(0) == returns.size(0) == advantages.size(0) == mask.size(0)
        assert observations.size(1) == actions.size(1) == returns.size(1) == advantages.size(1) == mask.size(1)

        return observations, actions, returns, advantages, mask

    data_loader = DataLoader(dataset, shuffle=True, drop_last=True, batch_size=batch_size, collate_fn=collate_fn)

    iteration = start_iteration

    for i in range(1, num_epochs + 1):
        if is_primary():
            logging.info(f"Starting epoch {i}")

        loss_sum = 0
        actor_loss_sum = 0
        critic_loss_sum = 0
        entropy_loss_sum = 0

        num_of_batches = 0

        for batch in data_loader:
            loss, actor_loss, critic_loss, entropy_loss, grad_norm = train_on_batch(
                device,
                optimizer,
                model,
                model_old,
                policy,
                batch,
                backpropagation_steps=backpropagation_steps,
                clip_range=clip_range,
                max_grad_norm=max_grad_norm,
                value_coefficient=value_coefficient,
                entropy_coefficient=entropy_coefficient
            )

            num_of_batches += 1

            if is_primary():
                loss_sum += loss
                actor_loss_sum += actor_loss
                critic_loss_sum += critic_loss
                entropy_loss_sum += entropy_loss
                metrics.add_scalar("Loss/Total", loss_sum / num_of_batches, iteration)
                metrics.add_scalar("Loss/Actor", actor_loss_sum / num_of_batches, iteration)
                metrics.add_scalar("Loss/Critic", critic_loss_sum / num_of_batches, iteration)
                metrics.add_scalar("Loss/Entropy", entropy_loss_sum / num_of_batches, iteration)
                metrics.add_scalar("Unclipped gradient norm", grad_norm, iteration)

            iteration += 1

        if is_primary():
            logging.info(f"Epoch total loss: {loss_sum / num_of_batches:.3f}, "
                         f"actor loss: {actor_loss_sum / num_of_batches:.3f}, "
                         f"critic loss: {critic_loss_sum / num_of_batches:.3f}, "
                         f"entropy loss: {entropy_loss_sum / num_of_batches:.3f}")

    return iteration


def compute_average_reward(trajectories):
    episode_rewards = [sum(trajectory.rewards) for trajectory in trajectories]

    return statistics.mean(episode_rewards)


def run_episodes_parallel(policy, envs, num_episodes):
    assert num_episodes % len(envs) == 0

    trajectories = []
    lock = threading.Lock()

    def run_episodes(env):
        data_collector = DataCollector(policy, env)

        for _ in range(num_episodes // len(envs)):
            trajectory = data_collector.run_episode()

            with lock:
                trajectories.append(trajectory)

    threads = []
    for env in envs:
        thread = Thread(target=run_episodes, args=(env,))
        threads.append(thread)
        thread.start()

    [thread.join() for thread in threads]

    assert len(trajectories) == num_episodes

    return trajectories


def collect_data_and_train(
        device,
        policy,
        envs,
        model,
        optimizer,
        metrics,
        iterations=1000,
        episodes_per_iteration=128,
        bootstrap_length=5,
        num_epochs=3,
        batch_size=128,
        backpropagation_steps=10,
        clip_range=0.2,
        max_grad_norm=5.0,
        value_coefficient=1.0,
        entropy_coefficient=0.01,
        evaluation_episodes=250
):
    assert episodes_per_iteration % len(envs) == 0
    assert evaluation_episodes % len(envs) == 0

    def critic(*args, **kwargs):
        return model(*args, **{**kwargs, **{"compute_policy_dists": False}})

    train_it = 0
    best_reward = -1e9

    for i in range(1, iterations + 1):
        if is_primary():
            logging.info(f"Starting iteration {i}")
        start_time = time.time()

        trajectories = run_episodes_parallel(policy, envs, episodes_per_iteration)
        worker_avg_reward = compute_average_reward(trajectories)
        avg_reward_sums = torch.tensor([worker_avg_reward], dtype=torch.float32, device=device)
        dist.reduce(avg_reward_sums, 0, op=ReduceOp.SUM)

        if is_primary():
            avg_reward = avg_reward_sums.item() / dist.get_world_size()
            logging.info(f"Average reward was {avg_reward:.3f}")
            metrics.add_scalar("Average reward", avg_reward, i)

            if avg_reward > best_reward:
                logging.info("Current average reward was higher than previous best, saving model")
                torch.save(model.state_dict(), "best_model.pt")
                best_reward = avg_reward

        logging.info("Data collection phase finished")

        dataset = create_dataset(device, critic, trajectories, bootstrap_length=bootstrap_length)

        train_it = train(
            device=device,
            dataset=dataset,
            model=model,
            policy=policy,
            optimizer=optimizer,
            metrics=metrics,
            start_iteration=train_it,
            num_epochs=num_epochs,
            batch_size=batch_size,
            backpropagation_steps=backpropagation_steps,
            clip_range=clip_range,
            max_grad_norm=max_grad_norm,
            value_coefficient=value_coefficient,
            entropy_coefficient=entropy_coefficient
        )

        if is_primary():
            torch.save(model.state_dict(), "last_model.pt")
            torch.save(optimizer.state_dict(), "last_optimizer.pt")

        end_time = time.time()
        seconds_elapsed = end_time - start_time
        if is_primary():
            logging.info(f"Iteration {i} finished, took total of {seconds_elapsed / 60:.1f} minutes")

    if is_primary():
        logging.info("Training finished")
        logging.info("Evaluating best model")

    evaluation_trajectories = run_episodes_parallel(policy, envs, evaluation_episodes)
    worker_eval_reward = compute_average_reward(evaluation_trajectories)
    eval_reward_sums = torch.tensor([worker_eval_reward], dtype=torch.float32)
    dist.reduce(eval_reward_sums, 0, op=ReduceOp.SUM)

    if is_primary():
        avg_eval_reward = eval_reward_sums.item() / dist.get_world_size()
        logging.info(f"Best model had average reward of {avg_eval_reward:.3f} per episode")

        return avg_eval_reward
