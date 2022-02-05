import math
import statistics

import pytest
import torch
from terminator import Action

from duels_training.sumo import action_log_probs, sample_action, compute_entropies, get_available_maps


def inverse_sigmoid(x):
    return -torch.log(1 / x - 1)


def test_action_log_prob_big_batch():
    batch_size = 17

    actions = Action(
        forward=torch.randint(low=0, high=3, size=[batch_size]) - 1,
        left=torch.randint(low=0, high=3, size=[batch_size]) - 1,
        jumping=torch.randint(low=0, high=2, size=[batch_size]).to(torch.bool),
        attacking=torch.randint(low=0, high=2, size=[batch_size]).to(torch.bool),
        sprinting=torch.randint(low=0, high=2, size=[batch_size]).to(torch.bool),
        delta_yaw=torch.randn([batch_size]) * 5,
        delta_pitch=torch.randn([batch_size]) * 3
    )

    model_output = torch.randn([batch_size, 13]) * 5

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
        inverse_sigmoid(torch.tensor([0.7, 0.95, 0.6])),
        inverse_sigmoid(torch.tensor([4.0 / 12 + 0.5, 3.0 / 6])),
        inverse_sigmoid(torch.tensor([-5.0 / 12 + 0.5, 2.0 / 6]))
    ], dim=0).unsqueeze(0)

    actions = Action(
        forward=torch.tensor([-1]),
        left=torch.tensor([0]),
        jumping=torch.tensor([False]),
        attacking=torch.tensor([True]),
        sprinting=torch.tensor([False]),
        delta_yaw=torch.tensor([5.5]),
        delta_pitch=torch.tensor([-4.0])
    )

    log_probs = action_log_probs(model_output=model_output, actions=actions)

    assert log_probs.shape == (1,)
    expected_log_prob = math.log(0.7 * 0.5 * 0.3 * 0.95 * 0.4 * 0.11736 * 0.17603)
    assert log_probs.item() == pytest.approx(expected_log_prob, abs=1e-4)


def test_compute_entropies():
    model_output = torch.cat([
        torch.log(torch.tensor([0.7, 0.2, 0.1])),
        torch.log(torch.tensor([0.1, 0.5, 0.4])),
        inverse_sigmoid(torch.tensor([0.7, 0.95, 0.6])),
        inverse_sigmoid(torch.tensor([4.0 / 12 + 0.5, 3.0 / 6])),
        inverse_sigmoid(torch.tensor([-5.0 / 12 + 0.5, 2.0 / 6]))
    ], dim=0).unsqueeze(0)

    entropy = compute_entropies(model_output)

    assert entropy.shape == (1,)

    expected_entropy = (0.8018 + 0.9433 + 0.6109 + 0.1985 + 0.6730 + 2.5176 + 2.1121) / 7
    assert entropy.item() == pytest.approx(expected_entropy, abs=1e-4)


def test_sample_action():
    model_output = torch.cat([
        torch.log(torch.tensor([0.7, 0.2, 0.1])),
        torch.log(torch.tensor([0.1, 0.5, 0.4])),
        inverse_sigmoid(torch.tensor([0.7, 0.95, 0.6])),
        inverse_sigmoid(torch.tensor([4.0 / 12 + 0.5, 3.0 / 6])),
        inverse_sigmoid(torch.tensor([-5.0 / 12 + 0.5, 2.0 / 6]))
    ], dim=0).unsqueeze(0)

    forward_values = []
    left_values = []
    jumping_values = []
    attacking_values = []
    sprinting_values = []
    delta_yaws = []
    delta_pitches = []

    for i in range(100):
        action = sample_action(model_output)

        forward_values.append(action.forward)
        left_values.append(action.left)
        jumping_values.append(action.jumping)
        attacking_values.append(action.attacking)
        sprinting_values.append(action.sprinting)
        delta_yaws.append(action.delta_yaw)
        delta_pitches.append(action.delta_pitch)

    assert len(list(filter(lambda x: x == -1, forward_values))) == pytest.approx(70, abs=5)
    assert len(list(filter(lambda x: x == 0, forward_values))) == pytest.approx(20, abs=5)
    assert len(list(filter(lambda x: x == 1, forward_values))) == pytest.approx(10, abs=5)

    assert len(list(filter(lambda x: x == -1, left_values))) == pytest.approx(10, abs=5)
    assert len(list(filter(lambda x: x == 0, left_values))) == pytest.approx(50, abs=5)
    assert len(list(filter(lambda x: x == 1, left_values))) == pytest.approx(40, abs=5)

    assert len(list(filter(lambda x: x is True, jumping_values))) == pytest.approx(70, abs=5)
    assert len(list(filter(lambda x: x is False, jumping_values))) == pytest.approx(30, abs=5)

    assert len(list(filter(lambda x: x is True, attacking_values))) == pytest.approx(95, abs=3)
    assert len(list(filter(lambda x: x is False, attacking_values))) == pytest.approx(5, abs=3)

    assert len(list(filter(lambda x: x is True, sprinting_values))) == pytest.approx(60, abs=10)
    assert len(list(filter(lambda x: x is False, sprinting_values))) == pytest.approx(40, abs=10)

    assert statistics.mean(delta_yaws) == pytest.approx(4, abs=0.1)
    assert statistics.mean(delta_pitches) == pytest.approx(-5, abs=0.1)


def test_get_available_maps():
    assert get_available_maps(0) == ["white_crystal"]
    assert get_available_maps(105) == ["white_crystal"]
    assert get_available_maps(199) == ["white_crystal"]
    assert get_available_maps(200) == ["white_crystal"]
    assert get_available_maps(201) == ["white_crystal"]
    assert get_available_maps(300) == ["white_crystal", "classic_sumo"]
    assert get_available_maps(317) == ["white_crystal", "classic_sumo"]
    assert get_available_maps(350) == ["white_crystal", "classic_sumo"]
    assert get_available_maps(400) == ["white_crystal", "classic_sumo", "space_mine"]
    assert get_available_maps(500) == ["white_crystal", "classic_sumo", "space_mine", "ponsen"]
    assert get_available_maps(600) == ["white_crystal", "classic_sumo", "space_mine", "ponsen", "fort_royale"]
    assert get_available_maps(601) == ["white_crystal", "classic_sumo", "space_mine", "ponsen", "fort_royale"]
    assert get_available_maps(700) == ["white_crystal", "classic_sumo", "space_mine", "ponsen", "fort_royale"]
    assert get_available_maps(1200) == ["white_crystal", "classic_sumo", "space_mine", "ponsen", "fort_royale"]
