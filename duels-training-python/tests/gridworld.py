import torch

from tests.models import RecurrentModel, PolicyForRecurrentModel


class Gridworld:
    LEFT = torch.tensor([0])
    RIGHT = torch.tensor([1])
    UP = torch.tensor([2])
    DOWN = torch.tensor([3])

    ACTIONS = [LEFT, RIGHT, UP, DOWN]

    def __init__(self, size=5, max_steps=10000, device=torch.device("cpu")):
        self.x = 0.0
        self.y = 0.0
        self.size = size
        self.max_steps = max_steps
        self.device = device
        self.episode_running = False
        self.steps_taken = 0

    def step(self, action):
        assert self.episode_running

        if action == Gridworld.LEFT:
            dx = -1
            dy = 0
        elif action == Gridworld.RIGHT:
            dx = 1
            dy = 0
        elif action == Gridworld.UP:
            dx = 0
            dy = 1
        elif action == Gridworld.DOWN:
            dx = 0
            dy = -1
        else:
            raise ValueError("Invalid action")

        self.steps_taken += 1

        self.x = max(min(self.size - 1.0, self.x + dx), 0.0)
        self.y = max(min(self.size - 1.0, self.y + dy), 0.0)

        reached_goal = self.x == self.size - 1 and self.y == self.size - 1
        done = reached_goal or self.steps_taken > self.max_steps

        if done:
            self.episode_running = False

        observation = self._get_state()
        reward = 10 if reached_goal else -1
        debug = {}

        return observation, reward, done, debug

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.episode_running = True
        self.steps_taken = 0

        return self._get_state()

    def _get_state(self):
        return torch.tensor([self.x, self.y], dtype=torch.float32, device=self.device)


class PolicyState:
    def __init__(self, model, observation):
        self.model = model
        self.last_state = None
        self.policy_dist = None

        self.update(observation)

    def update(self, observation):
        policy_dists, _, self.last_state = self.model(
            observation.unsqueeze(0).unsqueeze(0),
            initial_state=self.last_state,
            compute_value_estimates=False
        )

        self.policy_dist = torch.exp(policy_dists.squeeze(0).squeeze(0))

    def sample_action(self):
        action = torch.multinomial(self.policy_dist, num_samples=1).item()
        return Gridworld.ACTIONS[action]


class GridworldPolicy(PolicyForRecurrentModel):
    def __init__(self, model):
        super().__init__()

        self.model = model

    def get_initial_state(self, observation):
        return PolicyState(self.model, observation)


class GridworldModel(RecurrentModel):
    def __init__(self):
        super().__init__(observation_size=2, num_of_actions=4)
