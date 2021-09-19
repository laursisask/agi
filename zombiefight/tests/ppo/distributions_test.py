import math

import pytest
import torch
from terminator import Action

from ppo.distributions import ModelOutputDistribution
from ppo.model import ModelOutput, ContinuousActionOutput


@pytest.fixture
def batch_size():
    return 73


@pytest.fixture
def model_output(batch_size):
    return ModelOutput(
        mouse=ContinuousActionOutput(means=torch.randn([batch_size, 2]), stds=torch.rand([batch_size, 2])),
        forward_movement=torch.randn([batch_size, 3]) * 4,
        strafe_movement=torch.randn([batch_size, 3]) * 2,
        jump=torch.randn([batch_size]) * 1.5,
        attack=torch.randn([batch_size]) * 1.1,
        value=torch.randn([batch_size]) * 10
    )


def inverse_sigmoid(x):
    return -torch.log(1 / x - 1)


def test_sample(model_output, batch_size):
    dist = ModelOutputDistribution(model_output)

    action = dist.sample()

    assert action.forward.shape == (batch_size,)
    assert action.forward.dtype == torch.int64
    assert set(action.forward.tolist()) == {-1, 0, 1}

    assert action.left.shape == (batch_size,)
    assert action.left.dtype == torch.int64
    assert set(action.left.tolist()) == {-1, 0, 1}

    assert action.jumping.shape == (batch_size,)
    assert action.jumping.dtype == torch.bool
    assert set(action.jumping.tolist()) == {False, True}

    assert action.attacking.shape == (batch_size,)
    assert action.attacking.dtype == torch.bool
    assert set(action.attacking.tolist()) == {False, True}

    assert action.delta_yaw.shape == (batch_size,)
    assert action.delta_yaw.dtype == torch.float32
    assert -5 < torch.min(action.delta_yaw) < 0
    assert 0 < torch.max(action.delta_yaw) < 5

    assert action.delta_pitch.shape == (batch_size,)
    assert action.delta_pitch.dtype == torch.float32
    assert -5 < torch.min(action.delta_pitch) < 0
    assert 0 < torch.max(action.delta_pitch) < 5


def test_entropy(model_output):
    dist = ModelOutputDistribution(model_output)

    assert 0 < dist.entropy() < 10


def test_log_prob_big_batch(model_output, batch_size):
    action = Action(
        forward=torch.randint(low=0, high=3, size=[batch_size]) - 1,
        left=torch.randint(low=0, high=3, size=[batch_size]) - 1,
        jumping=torch.randint(low=0, high=2, size=[batch_size]).to(torch.bool),
        attacking=torch.randint(low=0, high=2, size=[batch_size]).to(torch.bool),
        delta_yaw=torch.randn([batch_size]) * 5,
        delta_pitch=torch.randn([batch_size]) * 3
    )

    dist = ModelOutputDistribution(model_output)

    log_prob = dist.log_prob(action)
    assert log_prob.dtype == torch.float32
    assert log_prob.shape == (batch_size,)
    assert torch.all(log_prob < 0)


def test_log_prob_individual():
    model_output = ModelOutput(
        mouse=ContinuousActionOutput(means=torch.tensor([[-5.0, 10.0]]), stds=torch.tensor([[2.0, 3.0]])),
        forward_movement=torch.log(torch.tensor([[0.7, 0.2, 0.1]])),
        strafe_movement=torch.log(torch.tensor([[0.1, 0.5, 0.4]])),
        jump=inverse_sigmoid(torch.tensor([0.7])),
        attack=inverse_sigmoid(torch.tensor([0.95])),
        value=torch.tensor([30.0])
    )

    action = Action(
        forward=torch.tensor([-1]),
        left=torch.tensor([0]),
        jumping=torch.tensor([False]),
        attacking=torch.tensor([True]),
        delta_yaw=torch.tensor([10.5]),
        delta_pitch=torch.tensor([-4.0])
    )

    dist = ModelOutputDistribution(model_output)

    log_prob = dist.log_prob(action)

    assert log_prob.shape == (1,)
    expected_log_prob = math.log(0.7 * 0.5 * 0.3 * 0.95 * 0.17603 * 0.13115)
    assert log_prob.item() == pytest.approx(expected_log_prob, abs=1e-4)
