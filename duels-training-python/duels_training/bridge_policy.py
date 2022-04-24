import torch
import torch.nn.functional as F
from terminator import Action


def compute_log_prob_dists(model_output):
    batch_size = model_output.size(0)
    assert model_output.shape == (batch_size, 21)

    eps = 1e-7

    forward_dist = F.log_softmax(model_output.narrow(dim=1, start=0, length=3), dim=1)
    left_dist = F.log_softmax(model_output.narrow(dim=1, start=3, length=3), dim=1)

    jumping_dist, sprinting_dist = torch.log(
        torch.clamp(
            torch.sigmoid(model_output.narrow(dim=1, start=6, length=2)),
            min=eps,
            max=1 - eps
        )
    ).split(split_size=1, dim=1)

    jumping_dist = jumping_dist.squeeze(1)
    sprinting_dist = sprinting_dist.squeeze(1)
    assert jumping_dist.shape == (batch_size,)
    assert sprinting_dist.shape == (batch_size,)

    yaw_a, yaw_b, pitch_a, pitch_b = model_output.narrow(dim=1, start=8, length=4).split(split_size=1, dim=1)
    yaw_a = yaw_a.squeeze(1)
    yaw_b = yaw_b.squeeze(1)
    pitch_a = pitch_a.squeeze(1)
    pitch_b = pitch_b.squeeze(1)
    assert yaw_a.shape == (batch_size,)
    assert yaw_b.shape == (batch_size,)
    assert pitch_a.shape == (batch_size,)
    assert pitch_b.shape == (batch_size,)

    yaw_mean = (torch.sigmoid(yaw_a) - 0.5) * 24
    yaw_std = torch.clamp(torch.sigmoid(yaw_b) * 9, min=eps)

    pitch_mean = (torch.sigmoid(pitch_a) - 0.5) * 12
    pitch_std = torch.clamp(torch.sigmoid(pitch_b) * 6, min=eps)

    clicking_dist = F.log_softmax(model_output.narrow(dim=1, start=12, length=3), dim=1)

    hotbar_dist = F.log_softmax(model_output.narrow(dim=1, start=15, length=6), dim=1)

    return (forward_dist, left_dist, jumping_dist, sprinting_dist, clicking_dist,
            yaw_mean, yaw_std, pitch_mean, pitch_std, hotbar_dist)


@torch.no_grad()
def sample_action(model_output):
    (forward_dist, left_dist, jumping_dist, sprinting_dist, clicking_dist,
     yaw_mean, yaw_std, pitch_mean, pitch_std, hotbar_dist) = compute_log_prob_dists(model_output)

    forward_action, left_action = torch.multinomial(
        torch.exp(torch.cat([forward_dist, left_dist], dim=0)),
        num_samples=1,
        replacement=True
    )

    clicking_action = torch.multinomial(torch.exp(clicking_dist), num_samples=1, replacement=True)
    # 0 - attacking
    # 1 - using
    # 2 - nothing

    hotbar_slot = torch.multinomial(torch.exp(hotbar_dist), num_samples=1, replacement=True)

    jumping_action = torch.bernoulli(torch.exp(jumping_dist))
    sprinting_action = torch.bernoulli(torch.exp(sprinting_dist))

    normal_actions = torch.normal(
        mean=torch.stack([yaw_mean, pitch_mean]),
        std=torch.stack([yaw_std, pitch_std])
    )

    yaw_action = normal_actions[0]
    pitch_action = normal_actions[1]

    return Action(
        forward=forward_action.item() - 1,
        left=left_action.item() - 1,
        jumping=bool(jumping_action.to(torch.bool).item()),
        attacking=clicking_action.item() == 0,
        using=clicking_action.item() == 1,
        sprinting=bool(sprinting_action.to(torch.bool).item()),
        delta_yaw=yaw_action.item(),
        delta_pitch=pitch_action.item(),
        inventory_slot=hotbar_slot.item()
    )
