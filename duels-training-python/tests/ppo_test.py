import random
import statistics

import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

from duels_training.incremental_stats_calculator import IncrementalStatsCalculator
from duels_training.ppo import DataCollector, Trajectory, create_dataset, collect_data_and_train, \
    compute_gradients
from tests.gridworld import Gridworld, GridworldPolicy, GridworldModel
from tests.models import RecurrentModel, PolicyForRecurrentModel


def test_run_episode_random_policy():
    class PolicyState:
        def update(self, observation):
            pass

        def sample_action(self):
            return random.choice(Gridworld.ACTIONS)

    class Policy:
        def get_initial_state(self, observation):
            return PolicyState()

    data_collector = DataCollector(policy=Policy(), env=Gridworld(size=8))

    trajectory = data_collector.run_episode()

    assert isinstance(trajectory, Trajectory)
    assert len(trajectory) >= 14
    assert torch.equal(trajectory.observations[0], torch.tensor([0.0, 0.0]))

    # Final state should not be included in the trajectory
    assert not torch.equal(trajectory.observations[-1], torch.tensor([7.0, 7.0]))

    # All rewards except last one should be -1
    for r in trajectory.rewards[:-1]:
        assert r == -1

    assert trajectory.rewards[-1] == 10


def test_run_episode_policy_state():
    class PolicyState:
        def __init__(self, observation):
            self.observation = observation

        def update(self, observation):
            self.observation = observation

        def sample_action(self):
            # This policy moves first all the way to the right and then
            # to the top
            x, y = self.observation

            if x == 4:
                return Gridworld.UP
            else:
                return Gridworld.RIGHT

    class Policy:
        def get_initial_state(self, observation):
            return PolicyState(observation)

    data_collector = DataCollector(policy=Policy(), env=Gridworld(size=5))
    trajectory = data_collector.run_episode()

    assert len(trajectory) == 8
    assert trajectory.rewards == [-1] * 7 + [10]
    expected_observations = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([2.0, 0.0]),
        torch.tensor([3.0, 0.0]),
        torch.tensor([4.0, 0.0]),
        torch.tensor([4.0, 1.0]),
        torch.tensor([4.0, 2.0]),
        torch.tensor([4.0, 3.0])
    ]

    for i, expected in enumerate(expected_observations):
        assert torch.equal(expected, trajectory.observations[i])

    assert trajectory.actions == [Gridworld.RIGHT] * 4 + [Gridworld.UP] * 4


def test_create_dataset_single_trajectory():
    device = torch.device("cpu")

    observations_list = [torch.randn([8]) for _ in range(6)]
    actions_list = [torch.randn([2]) for _ in range(6)]

    critic_calls = []

    def critic_fn(x, initial_state):
        assert x.shape == (1, 3, 8)
        critic_calls.append((x, initial_state))

        return torch.full([x.size(0), x.size(1)], 0.2), torch.randn([x.size(0), x.size(1)])

    class MockRewardStats:
        def __init__(self):
            self.n = 50

        def std(self):
            return 10.0

    trajectories = [
        Trajectory(
            observations=observations_list,
            rewards=[3, 2, -5, 1, 0, 9],
            actions=actions_list,
            metadatas=[{}] * 6
        )
    ]

    dataset = create_dataset(
        device=device,
        critic=critic_fn,
        trajectories=trajectories,
        reward_stats=MockRewardStats(),
        batch_size=13,
        bootstrap_length=2,
        steps_per_call=3
    )

    assert len(dataset) == 1

    observations, actions, returns, advantages = dataset[0]

    for i in range(6):
        assert torch.equal(observations_list[i], observations[i])
        assert torch.equal(actions_list[i], actions[i])

    assert torch.allclose(returns, torch.tensor([0.7, -0.1, -0.2, 0.3, 0.9, 0.9], dtype=torch.float32))
    assert torch.allclose(advantages, torch.tensor([0.5, -0.3, -0.4, 0.1, 0.7, 0.7], dtype=torch.float32))

    assert len(critic_calls) == 2


def test_create_dataset_gridworld():
    device = torch.device("cpu")

    critic_calls = []

    def critic_fn(x, initial_state):
        assert x.size(0) in [7, 13]
        assert x.size(1) in [1, 2, 3]
        assert x.size(2) == 2

        critic_calls.append((x, initial_state))
        return torch.full([x.size(0), x.size(1)], 0.1), torch.randn([x.size(0), x.size(1)])

    class PolicyState:
        def update(self, observation):
            pass

        def sample_action(self):
            return random.choice(Gridworld.ACTIONS)

    class Policy:
        def get_initial_state(self, observation):
            return PolicyState()

    class MockRewardStats:
        def __init__(self):
            self.n = 50

        def std(self):
            return 1.0

    data_collector = DataCollector(policy=Policy(), env=Gridworld(size=8))

    trajectories = [data_collector.run_episode() for _ in range(137)]

    dataset = create_dataset(
        device=device,
        critic=critic_fn,
        trajectories=trajectories,
        reward_stats=MockRewardStats(),
        batch_size=13,
        bootstrap_length=2,
        steps_per_call=3
    )

    assert len(dataset) == 137

    trajectories_sorted = sorted(trajectories, key=len)

    for i, (observations, actions, returns, advantages) in enumerate(dataset):
        length = observations.size(0)

        assert observations.shape == (length, 2)
        assert actions.shape == (length, 1)
        assert returns.shape == (length,)
        assert advantages.shape == (length,)

        original_trajectory = trajectories_sorted[i]

        assert length == len(original_trajectory)
        assert torch.equal(torch.stack(original_trajectory.observations), observations)
        assert torch.equal(torch.stack(original_trajectory.actions), actions)

        assert returns[-1].item() == pytest.approx(10.0)
        assert returns[-2].item() == pytest.approx(9.0)

        for i in range(length - 2):
            assert returns[i].item() == pytest.approx(-1.9)

        assert advantages[-1].item() == pytest.approx(9.9)
        assert advantages[-2].item() == pytest.approx(8.9)

        for i in range(length - 2):
            assert advantages[i].item() == pytest.approx(-2)


def test_compute_gradients():
    model = RecurrentModel()
    policy = PolicyForRecurrentModel()

    lengths = torch.randint(1, 45, [13, 1])
    max_length = torch.max(lengths).item()

    lengths_expanded = lengths.repeat([1, max_length])

    observations = torch.randn([13, max_length, 8])
    actions = torch.randint(0, 4, [13, max_length, 1])
    returns = torch.randn([13, max_length])
    advantages = torch.randn([13, max_length]) * 7

    mask = torch.arange(max_length).reshape([1, max_length]).repeat([13, 1]) < lengths_expanded

    (actor_loss, critic_loss, entropy_loss, loss, num_of_samples,
     last_state, last_state_old) = compute_gradients(
        device=torch.device("cpu"),
        model=model,
        policy=policy,
        model_old=model,
        last_state=None,
        last_state_old=None,
        observations=observations,
        actions=actions,
        returns=returns,
        advantages=advantages,
        mask=mask,
        clip_range=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01
    )

    assert num_of_samples == torch.sum(mask).item()

    # Sanity check
    # Make sure that these values do not change during refactoring
    assert loss / num_of_samples == pytest.approx(0.6, abs=0.1)
    assert actor_loss / num_of_samples == pytest.approx(-0.02, abs=0.01)
    assert critic_loss / num_of_samples == pytest.approx(1.2, abs=0.1)
    assert entropy_loss / num_of_samples == pytest.approx(-1.4, abs=0.1)

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    assert grad_norm.item() / num_of_samples == pytest.approx(0.4, abs=0.1)


def test_compute_gradients_critic_loss_zero():
    model = RecurrentModel()
    policy = PolicyForRecurrentModel()

    lengths = torch.randint(1, 45, [13, 1])
    max_length = torch.max(lengths).item()

    lengths_expanded = lengths.repeat([1, max_length])

    observations = torch.randn([13, max_length, 8])
    actions = torch.randint(0, 4, [13, max_length, 1])

    advantages = torch.randn([13, max_length]) * 7

    mask = torch.arange(max_length).reshape([1, max_length]).repeat([13, 1]) < lengths_expanded

    returns = model(observations)[1].detach()
    assert returns.shape == (13, max_length)
    returns[~mask] = 100.0

    (actor_loss, critic_loss, entropy_loss, loss, num_of_samples,
     last_state, last_state_old) = compute_gradients(
        device=torch.device("cpu"),
        model=model,
        policy=policy,
        model_old=model,
        last_state=None,
        last_state_old=None,
        observations=observations,
        actions=actions,
        returns=returns,
        advantages=advantages,
        mask=mask,
        clip_range=0.2,
        value_coefficient=5.0,
        entropy_coefficient=0.01
    )

    assert critic_loss / num_of_samples == pytest.approx(0, abs=1e-10)
    assert loss / num_of_samples == pytest.approx(actor_loss / num_of_samples + 0.01 * entropy_loss / num_of_samples,
                                                  abs=1e-6)


def test_collect_data_and_train(tmp_path):
    torch.manual_seed(0)
    random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GridworldModel()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    policy = GridworldPolicy(model)
    envs = [Gridworld(device=device) for _ in range(4)]

    collect_data_and_train(
        device=device,
        model=model,
        policy=policy,
        envs=envs,
        optimizer=optimizer,
        metrics=SummaryWriter(log_dir=tmp_path),
        reward_stats=IncrementalStatsCalculator(),
        iterations=14,
        steps_per_iteration=4096,
        parallel_sequences=8,
        batch_size=64,
        value_coefficient=0.005,
        entropy_coefficient=0.01,
        clip_range=0.2,
        bootstrap_length=4,
        num_epochs=3,
        assets_dir=tmp_path
    )

    env = Gridworld(device=device)
    rewards = []

    for i in range(256):
        done = False
        observation = env.reset()
        policy_state = policy.get_initial_state(observation)
        episode_reward = 0

        while not done:
            action = policy_state.sample_action()
            observation, reward, done, _ = env.step(action)

            episode_reward += reward
            policy_state.update(observation)

        rewards.append(episode_reward)

    eval_reward = statistics.mean(rewards)
    assert eval_reward == pytest.approx(2.8, abs=0.2)
