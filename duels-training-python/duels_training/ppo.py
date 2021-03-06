import copy
import logging
import os
import threading
import time
from dataclasses import dataclass
from threading import Thread
from typing import List

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


@dataclass
class Trajectory:
    observations: List
    actions: List
    rewards: List
    metadatas: List

    def __len__(self):
        return len(self.observations)

    def __post_init__(self):
        assert len(self.observations) == len(self.actions) == len(self.rewards) == len(self.metadatas)


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
        metadatas = []

        while not done:
            action = policy_state.sample_action()

            next_observation, reward, done, metadata = self.env.step(action)
            cumulative_reward += reward

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            metadatas.append(metadata)

            policy_state.update(next_observation)
            observation = next_observation

            steps_done += 1

        end_time = time.time()
        seconds_elapsed = end_time - start_time

        logging.info(f"Episode ended with cumulative reward of {cumulative_reward:.2f}, "
                     f"took {seconds_elapsed:.2f} seconds, "
                     f"TPS was {steps_done / seconds_elapsed:.2f}")

        return Trajectory(observations=observations, actions=actions, rewards=rewards, metadatas=metadatas)


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


def compute_n_step_returns(value_estimates: torch.Tensor, rewards: List[List[float]], n: int,
                           discount_factor: float) -> torch.Tensor:
    returns = torch.zeros(value_estimates.shape)

    num_of_trajectories = len(rewards)
    for i in range(num_of_trajectories):
        trajectory_length = len(rewards[i])
        working_return = 0

        for t in reversed(range(trajectory_length)):
            if t + n < trajectory_length:
                working_return -= rewards[i][t + n] * (discount_factor ** (n - 1))

            working_return = working_return * discount_factor + rewards[i][t]

            if t + n < trajectory_length:
                ret = working_return + (discount_factor ** n) * value_estimates[i][t + n].item()
            else:
                ret = working_return

            returns[i][t] = ret

    return returns


def compute_gae(value_estimates: torch.Tensor, rewards: List[List[float]], discount_factor: float,
                gae_param: float) -> torch.Tensor:
    batch_size = value_estimates.size(0)
    max_sequence_len = value_estimates.size(1)

    rewards_tensor = pad_sequence([torch.tensor(trajectory) for trajectory in rewards], batch_first=True)
    assert rewards_tensor.shape == value_estimates.shape

    next_state_value = torch.concat([value_estimates[:, 1:], torch.zeros(batch_size, 1)], dim=1)
    assert next_state_value.shape == value_estimates.shape

    lengths = torch.tensor([len(trajectory) for trajectory in rewards], dtype=torch.int32)
    next_state_value_mask = (torch.arange(max_sequence_len).unsqueeze(0).expand(batch_size, -1) + 1 <
                             lengths.unsqueeze(-1).expand(-1, max_sequence_len))
    assert next_state_value_mask.shape == value_estimates.shape

    residuals = rewards_tensor + discount_factor * next_state_value * next_state_value_mask - value_estimates
    assert residuals.shape == value_estimates.shape

    gae = torch.zeros(value_estimates.shape)

    working_gae = torch.zeros(batch_size, dtype=torch.float32)
    for i in reversed(range(max_sequence_len)):
        mask = i < lengths
        working_gae = residuals[:, i] * mask + (gae_param * discount_factor) * working_gae

        gae[:, i] = working_gae

    return gae


@torch.no_grad()
def create_dataset(device, critic, trajectories, bootstrap_length, discount_factor, gae_param, reward_stats,
                   batch_size=64, steps_per_call=32):
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

        reward_std = reward_stats.std() + 1e-7
        normalized_rewards = [[reward / reward_std for reward in trajectory_rewards] for trajectory_rewards in rewards]

        returns = compute_n_step_returns(value_estimates, normalized_rewards, bootstrap_length, discount_factor)

        advantages = compute_gae(value_estimates, normalized_rewards, discount_factor, gae_param)

        for t_observations, t_actions, t_rewards, t_returns, t_advantages in zip(
                observations,
                actions,
                normalized_rewards,
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


def compute_gradients(device, model, policy, model_old, last_state, last_state_old, observations, actions, returns,
                      advantages, mask, clip_range, value_coefficient, entropy_coefficient):
    observations = observations.to(device)
    actions = actions.to(device)
    returns = returns.to(device)
    advantages = advantages.to(device)
    mask = mask.to(device)

    policy_dists, value_estimates, last_state = model(observations, initial_state=last_state)

    log_probs = policy.action_log_prob(
        policy_dists.flatten(start_dim=0, end_dim=1),
        actions.flatten(start_dim=0, end_dim=1)
    ).reshape([actions.size(0), actions.size(1)])
    assert log_probs.shape == advantages.shape

    with torch.no_grad():
        policy_dists_old, _, last_state_old = model_old(
            observations,
            initial_state=last_state_old,
            compute_value_estimates=False
        )

    log_probs_old = policy.action_log_prob(
        policy_dists_old.flatten(start_dim=0, end_dim=1),
        actions.flatten(start_dim=0, end_dim=1)
    ).reshape([actions.size(0), actions.size(1)])
    assert log_probs_old.shape == advantages.shape

    ratios = torch.exp(log_probs - log_probs_old)
    num_of_samples = torch.sum(mask)
    assert num_of_samples >= 1

    surrogate_objective = torch.sum(torch.min(
        ratios * advantages,
        torch.clamp(ratios, 1 - clip_range, 1 + clip_range) * advantages
    ) * mask)

    actor_loss = -surrogate_objective
    assert actor_loss.numel() == 1
    assert not actor_loss.isnan()

    critic_loss = torch.sum(F.mse_loss(value_estimates, returns, reduction="none") * mask)
    assert not critic_loss.isnan()
    assert critic_loss.numel() == 1

    entropies = policy.entropy(
        policy_dists.flatten(start_dim=0, end_dim=1)
    ).reshape([policy_dists.size(0), policy_dists.size(1)])
    assert entropies.shape == mask.shape

    entropy = torch.sum(entropies * mask)
    entropy_loss = -entropy
    assert not entropy_loss.isnan()
    assert entropy_loss.numel() == 1

    loss = actor_loss + value_coefficient * critic_loss + entropy_coefficient * entropy_loss
    loss.backward()

    assert loss < 1e6

    last_state.detach_()

    return (actor_loss.item(), critic_loss.item(), entropy_loss.item(), loss.item(), num_of_samples.item(), last_state,
            last_state_old)


def train(device, dataset, model, policy, optimizer, metrics, start_iteration, num_epochs, batch_size,
          parallel_sequences, backpropagation_steps, clip_range, max_grad_norm, value_coefficient, entropy_coefficient):
    logging.info(f"Training on {len(dataset)} trajectories")

    model_old = copy.deepcopy(model)

    def collate_fn(batch):
        # batch: list of tuples where each tuple is
        # (observations: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor, advantages: torch.Tensor)
        observations, actions, returns, advantages = zip(*batch)
        assert len(observations) == len(actions) == len(advantages)
        assert len(observations) == parallel_sequences

        lengths = torch.tensor([len(trajectory_observations) for trajectory_observations in observations],
                               dtype=torch.int64)
        max_length = torch.max(lengths).item()

        mask = torch.arange(max_length).reshape([1, max_length]).expand(len(observations), -1) < lengths.unsqueeze(-1)

        observations = pad_sequence(observations, batch_first=True)
        actions = pad_sequence(actions, batch_first=True)
        returns = pad_sequence(returns, batch_first=True)
        advantages = pad_sequence(advantages, batch_first=True)

        assert observations.size(0) == actions.size(0) == returns.size(0) == advantages.size(0) == mask.size(0)
        assert observations.size(1) == actions.size(1) == returns.size(1) == advantages.size(1) == mask.size(1)

        return observations, actions, returns, advantages, mask

    data_loader = DataLoader(dataset, shuffle=True, drop_last=True, batch_size=parallel_sequences,
                             collate_fn=collate_fn)

    iteration = start_iteration

    for i in range(1, num_epochs + 1):
        logging.info(f"Starting epoch {i}")

        loss_sum = 0
        actor_loss_sum = 0
        critic_loss_sum = 0
        entropy_loss_sum = 0
        num_of_samples_sum = 0
        updates_done = 0

        samples_since_update = 0

        for batch in data_loader:
            all_observations, all_actions, all_returns, all_advantages, all_mask = batch
            all_advantages = standardize(all_advantages, all_mask)

            last_state = None
            last_state_old = None

            for observations, actions, returns, advantages, mask in zip(
                    all_observations.split(backpropagation_steps, dim=1),
                    all_actions.split(backpropagation_steps, dim=1),
                    all_returns.split(backpropagation_steps, dim=1),
                    all_advantages.split(backpropagation_steps, dim=1),
                    all_mask.split(backpropagation_steps, dim=1)
            ):
                (actor_loss, critic_loss, entropy_loss, loss, num_of_samples,
                 last_state, last_state_old) = compute_gradients(
                    device=device,
                    model=model,
                    policy=policy,
                    model_old=model_old,
                    last_state=last_state,
                    last_state_old=last_state_old,
                    observations=observations,
                    actions=actions,
                    returns=returns,
                    advantages=advantages,
                    mask=mask,
                    clip_range=clip_range,
                    value_coefficient=value_coefficient,
                    entropy_coefficient=entropy_coefficient
                )

                samples_since_update += num_of_samples
                num_of_samples_sum += num_of_samples

                if samples_since_update >= batch_size:
                    for p in model.parameters():
                        p.grad = torch.div(p.grad, samples_since_update)

                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                    model.zero_grad()
                    samples_since_update = 0
                    updates_done += 1

                    metrics.add_scalar("Unclipped gradient norm", grad_norm.item(), iteration)

                loss_sum += loss
                actor_loss_sum += actor_loss
                critic_loss_sum += critic_loss
                entropy_loss_sum += entropy_loss

                metrics.add_scalar("Loss/Total", loss_sum / num_of_samples_sum, iteration)
                metrics.add_scalar("Loss/Actor", actor_loss_sum / num_of_samples_sum, iteration)
                metrics.add_scalar("Loss/Critic", critic_loss_sum / num_of_samples_sum, iteration)
                metrics.add_scalar("Loss/Entropy", entropy_loss_sum / num_of_samples_sum, iteration)

                iteration += 1

        logging.info(f"Epoch total loss: {loss_sum / num_of_samples_sum:.3f}, "
                     f"actor loss: {actor_loss_sum / num_of_samples_sum:.3f}, "
                     f"critic loss: {critic_loss_sum / num_of_samples_sum:.3f}, "
                     f"entropy loss: {entropy_loss_sum / num_of_samples_sum:.3f}")

    return iteration


def run_episodes_parallel(policy, envs, num_steps, min_episodes):
    trajectories = []
    steps_done = 0
    lock = threading.Lock()

    def run_episodes(env):
        nonlocal steps_done

        data_collector = DataCollector(policy, env)

        while True:
            with lock:
                if steps_done > num_steps and len(trajectories) > min_episodes:
                    break

            trajectory = data_collector.run_episode()

            with lock:
                steps_done += len(trajectory)
                trajectories.append(trajectory)

    threads = []
    for env in envs:
        thread = Thread(target=run_episodes, args=(env,), daemon=True)
        threads.append(thread)
        thread.start()

    [thread.join() for thread in threads]

    assert steps_done >= num_steps

    return trajectories


def collect_data_and_train_iteration(device, policy, envs, model, optimizer, metrics, steps_per_iteration,
                                     train_it, global_it, critic, bootstrap_length, discount_factor, gae_param,
                                     num_epochs, batch_size, parallel_sequences, backpropagation_steps, clip_range,
                                     max_grad_norm, value_coefficient, entropy_coefficient, assets_dir, callback_fn,
                                     reward_stats):
    trajectories = run_episodes_parallel(policy, envs, steps_per_iteration, parallel_sequences)

    logging.info("Data collection phase finished")

    if callback_fn is not None:
        callback_fn(global_it, trajectories)

    [reward_stats.append(reward) for trajectory in trajectories for reward in trajectory.rewards]

    dataset = create_dataset(device, critic, trajectories,
                             bootstrap_length=bootstrap_length,
                             discount_factor=discount_factor,
                             gae_param=gae_param,
                             reward_stats=reward_stats)
    del trajectories

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
        parallel_sequences=parallel_sequences,
        backpropagation_steps=backpropagation_steps,
        clip_range=clip_range,
        max_grad_norm=max_grad_norm,
        value_coefficient=value_coefficient,
        entropy_coefficient=entropy_coefficient
    )

    torch.save(model.state_dict(), os.path.join(assets_dir, "last_model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(assets_dir, "last_optimizer.pt"))

    return train_it


def collect_data_and_train(
        device,
        policy,
        envs,
        model,
        optimizer,
        metrics,
        reward_stats,
        iterations=1000,
        steps_per_iteration=4096,
        bootstrap_length=5,
        discount_factor=0.99,
        gae_param=0.95,
        num_epochs=3,
        parallel_sequences=128,
        batch_size=1280,
        backpropagation_steps=10,
        clip_range=0.2,
        max_grad_norm=5.0,
        value_coefficient=1.0,
        entropy_coefficient=0.01,
        post_iteration_callback=None,
        post_data_collect_callback=None,
        start_global_iteration=1,
        start_train_iteration=0,
        assets_dir="."
):
    def critic(*args, **kwargs):
        _, values, final_state = model(*args, **{**kwargs, **{"compute_policy_dists": False}})
        return values, final_state

    train_it = start_train_iteration

    for i in range(start_global_iteration, iterations + 1):
        logging.info(f"Starting iteration {i}")
        start_time = time.time()

        train_it = collect_data_and_train_iteration(
            device=device,
            policy=policy,
            envs=envs,
            model=model,
            optimizer=optimizer,
            metrics=metrics,
            steps_per_iteration=steps_per_iteration,
            train_it=train_it,
            global_it=i,
            critic=critic,
            bootstrap_length=bootstrap_length,
            discount_factor=discount_factor,
            gae_param=gae_param,
            num_epochs=num_epochs,
            batch_size=batch_size,
            parallel_sequences=parallel_sequences,
            backpropagation_steps=backpropagation_steps,
            clip_range=clip_range,
            max_grad_norm=max_grad_norm,
            value_coefficient=value_coefficient,
            entropy_coefficient=entropy_coefficient,
            assets_dir=assets_dir,
            callback_fn=post_data_collect_callback,
            reward_stats=reward_stats
        )

        end_time = time.time()
        seconds_elapsed = end_time - start_time
        logging.info(f"Iteration {i} finished, took total of {seconds_elapsed / 60:.1f} minutes")

        if post_iteration_callback is not None:
            post_iteration_callback(i, train_it)

        metrics.add_scalar("Standard deviation of reward", reward_stats.std(), i)

    logging.info("Training finished")
    logging.info("Evaluating best model")
