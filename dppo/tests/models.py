import torch
import torch.nn.functional as F


class RecurrentModel(torch.nn.Module):
    def __init__(self, observation_size=8, num_of_actions=4):
        super().__init__()

        self.rnn = torch.nn.LSTM(observation_size, 12, batch_first=True)
        self.policy_head = torch.nn.Linear(12, num_of_actions)
        self.value_head = torch.nn.Linear(12, 1)

    def forward(self, x, initial_state=None, compute_value_estimates=True, compute_policy_dists=True):
        if initial_state is not None:
            initial_state = (initial_state[0], initial_state[1])

        x, final_state = self.rnn(x, initial_state)

        if compute_policy_dists:
            policy_dists = F.log_softmax(self.policy_head(x), dim=-1)

        if compute_value_estimates:
            value_estimates = self.value_head(x).squeeze(-1)

        final_state = torch.stack(final_state)

        if compute_value_estimates and compute_policy_dists:
            return policy_dists, value_estimates, final_state
        elif compute_policy_dists:
            return policy_dists, final_state
        elif compute_value_estimates:
            return value_estimates, final_state
        else:
            raise ValueError("Should compute at least one of value and policy distributions")


class PolicyForRecurrentModel:
    def action_log_prob(self, policy_dists, actions):
        return torch.gather(policy_dists, index=actions, dim=-1).squeeze(-1)

    def entropy(self, policy_dists):
        min_real = torch.finfo(policy_dists.dtype).min
        logits = torch.clamp(policy_dists, min=min_real)
        p_log_p = logits * F.softmax(logits, dim=-1)
        return -p_log_p.sum(dim=-1)
