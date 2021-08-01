import torch


class StackedStateConstructor:
    def __init__(self, stack_size):
        self.states = []
        self.stack_size = stack_size

    def current(self):
        assert len(self.states) >= self.stack_size

        images = torch.stack(list(map(lambda s: s.image, self.states[-self.stack_size:])), dim=0)
        others = self.states[-1].others

        return images, others

    def append(self, state):
        if self.states:
            self.states.append(state)
        else:
            self.states.extend([state] * self.stack_size)

    def clear(self):
        self.states.clear()
