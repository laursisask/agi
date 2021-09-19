from terminator import Action


def to_python_types(action: Action):
    return apply_to_components(lambda x: x.item(), action)


def to_device(action: Action, device):
    return apply_to_components(lambda x: x.to(device), action)


def apply_to_components(func, action: Action):
    return Action(
        forward=func(action.forward),
        left=func(action.left),
        jumping=func(action.jumping),
        attacking=func(action.attacking),
        delta_yaw=func(action.delta_yaw),
        delta_pitch=func(action.delta_pitch)
    )
