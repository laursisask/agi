import torch

from ppo.model import PPOModel


def test_model():
    model = PPOModel()

    batch_size = 11
    images = torch.randn([batch_size, 4, 84, 84])
    others = torch.randn([batch_size, 7])

    output = model(images=images, others=others)

    assert output.mouse.means.shape == (batch_size, 2)

    assert output.mouse.stds.shape == (batch_size, 2)
    assert torch.all(output.mouse.stds > 0)

    assert output.forward_movement.shape == (batch_size, 3)

    assert output.strafe_movement.shape == (batch_size, 3)

    assert output.jump.shape == (batch_size,)

    assert output.attack.shape == (batch_size,)

    assert output.value.shape == (batch_size,)


def test_result_independent_of_batch():
    torch.manual_seed(1)
    model = PPOModel()

    batch_size = 67
    images = torch.randn([batch_size, 4, 84, 84])
    others = torch.randn([batch_size, 7])

    batched_results = model(images=images, others=others)

    for i in range(batch_size):
        output = model(images=images[i].unsqueeze(0), others=others[i].unsqueeze(0))

        assert torch.allclose(output.mouse.means.squeeze(0), batched_results.mouse.means[i], atol=1e-6)
        assert torch.allclose(output.mouse.stds.squeeze(0), batched_results.mouse.stds[i], atol=1e-6)

        assert torch.allclose(output.forward_movement.squeeze(0), batched_results.forward_movement[i], atol=1e-6)
        assert torch.allclose(output.strafe_movement.squeeze(0), batched_results.strafe_movement[i], atol=1e-6)
        assert torch.allclose(output.attack.squeeze(0), batched_results.attack[i], atol=1e-6)
        assert torch.allclose(output.jump.squeeze(0), batched_results.jump[i], atol=1e-6)
        assert torch.allclose(output.value.squeeze(0), batched_results.value[i], atol=1e-6)
