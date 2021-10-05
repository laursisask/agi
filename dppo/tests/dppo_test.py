import random

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from dppo.dppo import DataCollector, Trajectory, create_dataset, train_on_batch, collect_data_and_train
from tests.gridworld import Gridworld, GridworldPolicy, GridworldModel
from tests.models import RecurrentModel, PolicyForRecurrentModel


def set_up_process_group(tmp_path):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmp_path}/sharedfile",
        rank=0,
        world_size=1
    )


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

        return torch.full([x.size(0), x.size(1)], 2.0), torch.randn([x.size(0), x.size(1)])

    trajectories = [
        Trajectory(
            observations=observations_list,
            rewards=[3, 2, -5, 1, 0, 9],
            actions=actions_list
        )
    ]

    dataset = create_dataset(
        device=device,
        critic=critic_fn,
        trajectories=trajectories,
        batch_size=13,
        bootstrap_length=2,
        steps_per_call=3
    )

    assert len(dataset) == 1

    observations, actions, returns, advantages = dataset[0]

    for i in range(6):
        assert torch.equal(observations_list[i], observations[i])
        assert torch.equal(actions_list[i], actions[i])

    assert torch.equal(returns, torch.tensor([7, -1, -2, 3, 9, 9], dtype=torch.float32))
    assert torch.equal(advantages, torch.tensor([5, -3, -4, 1, 7, 7], dtype=torch.float32))

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

    data_collector = DataCollector(policy=Policy(), env=Gridworld(size=8))

    trajectories = [data_collector.run_episode() for _ in range(137)]

    dataset = create_dataset(
        device=device,
        critic=critic_fn,
        trajectories=trajectories,
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

        assert returns[-1] == 10.0
        assert returns[-2] == 9.0

        for i in range(length - 2):
            assert returns[i] == -1.9

        assert advantages[-1] == 9.9
        assert advantages[-2] == 8.9

        for i in range(length - 2):
            assert advantages[i] == -2


def test_train_on_batch(tmp_path):
    set_up_process_group(tmp_path)

    model = RecurrentModel()
    policy = PolicyForRecurrentModel()
    optimizer = torch.optim.Adam(model.parameters())

    lengths = torch.randint(1, 45, [13, 1])
    max_length = torch.max(lengths).item()

    lengths_expanded = lengths.repeat([1, max_length])

    observations = torch.randn([13, max_length, 8])
    actions = torch.randint(0, 4, [13, max_length, 1])
    returns = torch.randn([13, max_length])
    advantages = torch.randn([13, max_length]) * 7

    mask = torch.arange(max_length).reshape([1, max_length]).repeat([13, 1]) < lengths_expanded

    batch = (observations, actions, returns, advantages, mask)

    loss, actor_loss, critic_loss, entropy_loss, grad_norm = train_on_batch(
        torch.device("cpu"),
        optimizer,
        model,
        model,
        policy,
        batch,
        backpropagation_steps=10,
        clip_range=0.2,
        max_grad_norm=5.0,
        value_coefficient=0.5,
        entropy_coefficient=0.01
    )

    # Sanity check
    # Make sure that these values do not change during refactoring
    assert loss == pytest.approx(0.6, abs=0.1)
    assert actor_loss == pytest.approx(0.0003, abs=0.0001)
    assert critic_loss == pytest.approx(1.2, abs=0.1)
    assert entropy_loss == pytest.approx(-1.4, abs=0.1)
    assert grad_norm == pytest.approx(0.4, abs=0.1)

    dist.destroy_process_group()


def test_train_on_batch_critic_loss_zero(tmp_path):
    set_up_process_group(tmp_path)

    model = RecurrentModel()
    policy = PolicyForRecurrentModel()
    optimizer = torch.optim.Adam(model.parameters())

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

    batch = (observations, actions, returns, advantages, mask)

    loss, actor_loss, critic_loss, entropy_loss, grad_norm = train_on_batch(
        torch.device("cpu"),
        optimizer,
        model,
        model,
        policy,
        batch,
        backpropagation_steps=10,
        clip_range=0.2,
        max_grad_norm=5.0,
        value_coefficient=0.5,
        entropy_coefficient=0.01
    )

    assert critic_loss == pytest.approx(0, abs=1e-10)
    assert loss == pytest.approx(actor_loss + 0.01 * entropy_loss, abs=1e-6)

    dist.destroy_process_group()


def launch_worker(rank, world_size, tmp_path):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmp_path}/sharedfile",
        rank=rank,
        world_size=world_size
    )

    torch.manual_seed(rank)
    random.seed(rank)

    model = GridworldModel()
    optimizer = torch.optim.Adam(model.parameters())
    device = torch.device("cpu")
    policy = GridworldPolicy(model)
    envs = [Gridworld() for _ in range(4)]

    eval_reward = collect_data_and_train(
        device=device,
        model=model,
        policy=policy,
        envs=envs,
        optimizer=optimizer if rank == 0 else None,
        metrics=SummaryWriter(log_dir=tmp_path) if rank == 0 else None,
        iterations=50,
        episodes_per_iteration=64,
        batch_size=8,
        value_coefficient=0.005,
        entropy_coefficient=0.01,
        clip_range=0.2,
        bootstrap_length=4,
        num_epochs=3,
        evaluation_episodes=256
    )

    if rank == 0:
        assert eval_reward == pytest.approx(2.5, abs=0.3)


def test_collect_data_and_train(tmp_path):
    nprocs = 3
    mp.spawn(launch_worker, args=(nprocs, tmp_path), nprocs=nprocs)

# TODO: Test some other gridworld where game state is not Markov
