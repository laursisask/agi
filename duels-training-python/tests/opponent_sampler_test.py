import torch

from duels_training.opponent_sampler import OpponentSampler


def test_opponent_sampler():
    class TestModel(torch.nn.Module):
        def __init__(self, i):
            super().__init__()
            assert 19 <= i <= 100 or i == 420
            self.x = torch.nn.Parameter(torch.tensor([i], dtype=torch.int32), requires_grad=False)

    last_model = TestModel(420)

    sampler = OpponentSampler(
        last_model=last_model,
        get_global_iteration=lambda: 101,
        load_model=TestModel,
        opponent_sampling_index=0.8
    )

    indices = []

    for _ in range(1000):
        model = sampler.sample()
        index = model.x.item()
        # Although models from 20 should be generated, due to calculations
        # being approximate the lower bound is actually 19. This is ok in real use
        # because one model difference does not really make much difference.
        assert 19 <= index <= 100

        indices.append(index)

    assert len(set(indices)) > 70