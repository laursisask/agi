import torch
from terminator import Action

from ppo.model import ModelOutput


class ModelOutputDistribution:
    forward: torch.distributions.Distribution
    strafe: torch.distributions.Distribution
    jumping: torch.distributions.Distribution
    attacking: torch.distributions.Distribution
    delta_yaw: torch.distributions.Distribution
    delta_pitch: torch.distributions.Distribution

    def __init__(self, model_output: ModelOutput):
        self.forward = torch.distributions.categorical.Categorical(logits=model_output.forward_movement)
        self.strafe = torch.distributions.categorical.Categorical(logits=model_output.strafe_movement)

        self.jumping = torch.distributions.bernoulli.Bernoulli(logits=model_output.jump)
        self.attacking = torch.distributions.bernoulli.Bernoulli(logits=model_output.attack)

        self.delta_pitch = torch.distributions.normal.Normal(
            loc=model_output.mouse.means[:, 0],
            scale=model_output.mouse.stds[:, 0]
        )

        self.delta_yaw = torch.distributions.normal.Normal(
            loc=model_output.mouse.means[:, 1],
            scale=model_output.mouse.stds[:, 1]
        )

    def sample(self):
        return Action(
            forward=self.forward.sample() - 1,
            left=self.strafe.sample() - 1,
            jumping=self.jumping.sample().to(torch.bool),
            attacking=self.attacking.sample().to(torch.bool),
            delta_yaw=self.delta_yaw.sample(),
            delta_pitch=self.delta_pitch.sample()
        )

    def entropy(self):
        # Not actual entropy but still a good estimate for keeping the model
        # from converging on a sub-optimal policy

        return (self.forward.entropy() + self.strafe.entropy() + self.jumping.entropy() + self.attacking.entropy()
                + self.delta_yaw.entropy() + self.delta_pitch.entropy()).mean()

    def log_prob(self, action: Action):
        batch_size = self.forward.batch_shape[0]
        assert batch_size == action.forward.size(0)

        forward_prob = self.forward.log_prob(action.forward + 1)
        assert forward_prob.shape == (batch_size,)

        strafe_prob = self.strafe.log_prob(action.left + 1)
        assert strafe_prob.shape == (batch_size,)

        jumping_prob = self.jumping.log_prob(action.jumping.to(torch.float32))
        assert jumping_prob.shape == (batch_size,)

        attacking_prob = self.attacking.log_prob(action.attacking.to(torch.float32))
        assert attacking_prob.shape == (batch_size,)

        delta_yaw_prob = self.delta_yaw.log_prob(action.delta_yaw)
        assert delta_yaw_prob.shape == (batch_size,)

        delta_pitch_prob = self.delta_pitch.log_prob(action.delta_pitch)
        assert delta_pitch_prob.shape == (batch_size,)

        return forward_prob + strafe_prob + jumping_prob + attacking_prob + delta_yaw_prob + delta_pitch_prob
