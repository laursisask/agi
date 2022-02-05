import torchvision.transforms.functional as TF


def transform_raw_state(state):
    assert (84, 84) == state.camera.shape
    return TF.normalize(state.camera.float().unsqueeze(0), [128], [256])
