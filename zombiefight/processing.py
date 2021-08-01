from collections import namedtuple

import torch
import torchvision.transforms.functional as TF

Observation = namedtuple("Observation", ["image", "others"])


def create_observation(obs):
    state = obs.camera

    assert (336, 336, 3) == state.shape

    permuted_image = state.permute(2, 0, 1)
    assert permuted_image.shape == (3, 336, 336)

    grayscaled_image = TF.rgb_to_grayscale(permuted_image)
    assert grayscaled_image.shape == (1, 336, 336)

    resized_image = TF.resize(grayscaled_image, [84, 84])
    assert resized_image.shape == (1, 84, 84)

    normalized_image = TF.normalize(resized_image.float(), [128], [256])

    final_image = normalized_image.squeeze(0)
    assert final_image.shape == (84, 84)

    others = torch.tensor([
        obs.zombie_x / 16,
        obs.zombie_y / 16,
        obs.zombie_z / 16,
        obs.pitch / 90,
        ((obs.yaw % 360) - 180) / 180,
        (obs.player_health - 10) / 10,
        (obs.zombie_health - 10) / 10
    ], dtype=torch.float32)

    return Observation(image=final_image, others=others)
