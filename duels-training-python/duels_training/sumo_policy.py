from typing import Tuple

import terminator
import torch
import torch.nn.functional as F
from terminator import Action


@torch.jit.script
def compute_log_prob_dists(model_output: torch.Tensor) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    batch_size: int = model_output.size(0)
    assert model_output.shape == (batch_size, 13)

    eps: float = 1e-7

    forward_dist = F.log_softmax(model_output.narrow(dim=1, start=0, length=3), dim=1)
    left_dist = F.log_softmax(model_output.narrow(dim=1, start=3, length=3), dim=1)

    jumping_dist, attacking_dist, sprinting_dist = torch.log(
        torch.clamp(
            torch.sigmoid(model_output.narrow(dim=1, start=6, length=3)),
            min=eps,
            max=1 - eps
        )
    ).split(split_size=1, dim=1)

    jumping_dist = jumping_dist.squeeze(1)
    attacking_dist = attacking_dist.squeeze(1)
    sprinting_dist = sprinting_dist.squeeze(1)
    assert jumping_dist.shape == (batch_size,)
    assert attacking_dist.shape == (batch_size,)
    assert sprinting_dist.shape == (batch_size,)

    yaw_a, yaw_b, pitch_a, pitch_b = model_output.narrow(dim=1, start=9, length=4).split(split_size=1, dim=1)
    yaw_a = yaw_a.squeeze(1)
    yaw_b = yaw_b.squeeze(1)
    pitch_a = pitch_a.squeeze(1)
    pitch_b = pitch_b.squeeze(1)
    assert yaw_a.shape == (batch_size,)
    assert yaw_b.shape == (batch_size,)
    assert pitch_a.shape == (batch_size,)
    assert pitch_b.shape == (batch_size,)

    yaw_mean = (torch.sigmoid(yaw_a) - 0.5) * 12
    yaw_std = torch.clamp(torch.sigmoid(yaw_b) * 6, min=eps)

    pitch_mean = (torch.sigmoid(pitch_a) - 0.5) * 12
    pitch_std = torch.clamp(torch.sigmoid(pitch_b) * 6, min=eps)

    return (forward_dist, left_dist, jumping_dist, attacking_dist, sprinting_dist,
            yaw_mean, yaw_std, pitch_mean, pitch_std)


@torch.no_grad()
def sample_action(model_output: torch.Tensor) -> terminator.Action:
    (forward_dist, left_dist, jumping_dist, attacking_dist, sprinting_dist,
     yaw_mean, yaw_std, pitch_mean, pitch_std) = compute_log_prob_dists(model_output)

    forward_action = torch.multinomial(torch.exp(forward_dist), num_samples=1)
    left_action = torch.multinomial(torch.exp(left_dist), num_samples=1)

    jumping_action = torch.bernoulli(torch.exp(jumping_dist))
    attacking_action = torch.bernoulli(torch.exp(attacking_dist))
    sprinting_action = torch.bernoulli(torch.exp(sprinting_dist))

    normal_actions = torch.normal(
        mean=torch.stack([yaw_mean, pitch_mean]),
        std=torch.stack([yaw_std, pitch_std])
    )

    yaw_action = normal_actions[0]
    pitch_action = normal_actions[1]

    return Action(
        forward_action.item() - 1,
        left_action.item() - 1,
        bool(jumping_action.to(torch.bool).item()),
        bool(attacking_action.to(torch.bool).item()),
        bool(sprinting_action.to(torch.bool).item()),
        yaw_action.item(),
        pitch_action.item()
    )
