import math
import statistics

import pytest
import torch
from terminator import Action

from duels_training.classic import action_log_probs, sample_action, compute_entropies


def inverse_sigmoid(x):
    return -torch.log(1 / x - 1)


def test_action_log_prob_big_batch():
    batch_size = 17

    clicking = torch.randint(low=0, high=2, size=[batch_size])

    actions = Action(
        forward=torch.randint(low=0, high=3, size=[batch_size]) - 1,
        left=torch.randint(low=0, high=3, size=[batch_size]) - 1,
        jumping=torch.randint(low=0, high=2, size=[batch_size]).to(torch.bool),
        attacking=clicking == 0,
        using=clicking == 1,
        sprinting=torch.randint(low=0, high=2, size=[batch_size]).to(torch.bool),
        delta_yaw=torch.randn([batch_size]) * 5,
        delta_pitch=torch.randn([batch_size]) * 3,
        inventory_slot=torch.randint(low=0, high=2, size=[batch_size])
    )

    model_output = torch.randn([batch_size, 18]) * 5

    log_probs = action_log_probs(model_output=model_output, actions=actions)

    assert log_probs.dtype == torch.float32
    assert log_probs.shape == (batch_size,)
    assert not log_probs.isnan().any()
    assert not log_probs.isinf().any()
    assert torch.all(log_probs < 0)


def test_action_log_prob_individual():
    model_output = torch.cat([
        torch.log(torch.tensor([0.7, 0.2, 0.1])),
        torch.log(torch.tensor([0.1, 0.5, 0.4])),
        inverse_sigmoid(torch.tensor([0.7, 0.95])),
        inverse_sigmoid(torch.tensor([4.0 / 24 + 0.5, 3.0 / 9])),
        inverse_sigmoid(torch.tensor([-5.0 / 12 + 0.5, 2.0 / 6])),
        torch.log(torch.tensor([0.4, 0.5, 0.1])),
        torch.log(torch.tensor([0.35, 0.55, 0.1]))
    ], dim=0).unsqueeze(0).expand(3, -1)

    actions = Action(
        forward=torch.tensor([-1, -1, -1]),
        left=torch.tensor([0, 0, 0]),
        jumping=torch.tensor([False, False, False]),
        attacking=torch.tensor([True, False, False]),
        using=torch.tensor([False, True, False]),
        sprinting=torch.tensor([False, False, False]),
        delta_yaw=torch.tensor([5.5, 5.5, 5.5]),
        delta_pitch=torch.tensor([-4.0, -4.0, -4.0]),
        inventory_slot=torch.tensor([1, 1, 1])
    )

    log_probs = action_log_probs(model_output=model_output, actions=actions)

    assert log_probs.shape == (3,)

    expected_log_prob1 = math.log(0.7 * 0.5 * 0.3 * 0.05 * 0.11736 * 0.17603 * 0.4 * 0.55)
    assert log_probs[0].item() == pytest.approx(expected_log_prob1, abs=1e-4)

    expected_log_prob2 = math.log(0.7 * 0.5 * 0.3 * 0.05 * 0.11736 * 0.17603 * 0.5 * 0.55)
    assert log_probs[1].item() == pytest.approx(expected_log_prob2, abs=1e-4)

    expected_log_prob3 = math.log(0.7 * 0.5 * 0.3 * 0.05 * 0.11736 * 0.17603 * 0.1 * 0.55)
    assert log_probs[2].item() == pytest.approx(expected_log_prob3, abs=1e-4)


def test_compute_entropies():
    model_output = torch.cat([
        torch.log(torch.tensor([0.7, 0.2, 0.1])),
        torch.log(torch.tensor([0.1, 0.5, 0.4])),
        inverse_sigmoid(torch.tensor([0.7, 0.95])),
        inverse_sigmoid(torch.tensor([4.0 / 24 + 0.5, 3.0 / 9])),
        inverse_sigmoid(torch.tensor([-5.0 / 12 + 0.5, 2.0 / 6])),
        torch.log(torch.tensor([0.4, 0.5, 0.1])),
        torch.log(torch.tensor([0.35, 0.55, 0.1]))
    ], dim=0).unsqueeze(0)

    entropy = compute_entropies(model_output)

    assert entropy.shape == (1,)

    expected_entropy = (0.8018 + 0.9433 + 0.6109 + 0.1985 + 2.5176 + 2.1121 + 0.9433 + 0.9265) / 8
    assert entropy.item() == pytest.approx(expected_entropy, abs=1e-4)


def test_sample_action():
    model_output = torch.cat([
        torch.log(torch.tensor([0.7, 0.2, 0.1])),
        torch.log(torch.tensor([0.1, 0.5, 0.4])),
        inverse_sigmoid(torch.tensor([0.7, 0.95])),
        inverse_sigmoid(torch.tensor([4.0 / 24 + 0.5, 3.0 / 9])),
        inverse_sigmoid(torch.tensor([-5.0 / 12 + 0.5, 2.0 / 6])),
        torch.log(torch.tensor([0.4, 0.5, 0.1])),
        torch.log(torch.tensor([0.35, 0.55, 0.1]))
    ], dim=0).unsqueeze(0)

    forward_values = []
    left_values = []
    jumping_values = []
    attacking_values = []
    using_values = []
    sprinting_values = []
    delta_yaws = []
    delta_pitches = []
    inventory_slots = []

    for i in range(1000):
        action = sample_action(model_output)

        forward_values.append(action.forward)
        left_values.append(action.left)
        jumping_values.append(action.jumping)
        attacking_values.append(action.attacking)
        using_values.append(action.using)
        sprinting_values.append(action.sprinting)
        delta_yaws.append(action.delta_yaw)
        delta_pitches.append(action.delta_pitch)
        inventory_slots.append(action.inventory_slot)

    assert len(list(filter(lambda x: x == -1, forward_values))) == pytest.approx(700, abs=50)
    assert len(list(filter(lambda x: x == 0, forward_values))) == pytest.approx(200, abs=50)
    assert len(list(filter(lambda x: x == 1, forward_values))) == pytest.approx(100, abs=50)

    assert len(list(filter(lambda x: x == -1, left_values))) == pytest.approx(100, abs=50)
    assert len(list(filter(lambda x: x == 0, left_values))) == pytest.approx(500, abs=50)
    assert len(list(filter(lambda x: x == 1, left_values))) == pytest.approx(400, abs=50)

    assert len(list(filter(lambda x: x is True, jumping_values))) == pytest.approx(700, abs=50)
    assert len(list(filter(lambda x: x is False, jumping_values))) == pytest.approx(300, abs=50)

    assert len(list(filter(lambda x: x is True, attacking_values))) == pytest.approx(400, abs=50)
    assert len(list(filter(lambda x: x is False, attacking_values))) == pytest.approx(600, abs=50)

    assert len(list(filter(lambda x: x is True, using_values))) == pytest.approx(500, abs=50)
    assert len(list(filter(lambda x: x is False, using_values))) == pytest.approx(500, abs=50)

    assert len(list(filter(lambda x: x is True, sprinting_values))) == pytest.approx(950, abs=30)
    assert len(list(filter(lambda x: x is False, sprinting_values))) == pytest.approx(50, abs=30)

    assert len(list(filter(lambda x: x == 0, inventory_slots))) == pytest.approx(350, abs=50)
    assert len(list(filter(lambda x: x == 1, inventory_slots))) == pytest.approx(550, abs=50)
    assert len(list(filter(lambda x: x == 2, inventory_slots))) == pytest.approx(100, abs=50)

    assert statistics.mean(delta_yaws) == pytest.approx(4, abs=0.1)
    assert statistics.mean(delta_pitches) == pytest.approx(-5, abs=0.1)
