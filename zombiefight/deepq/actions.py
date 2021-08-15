from terminator import Action

ACTIONS = [
    # Do nothing:
    Action(),
    # Do one thing at a time:
    Action(forward=1),
    Action(forward=-1),
    Action(left=1),
    Action(left=-1),
    Action(jumping=True),
    Action(attacking=True),
    Action(delta_pitch=9),
    Action(delta_pitch=-9),
    Action(delta_yaw=18),
    Action(delta_yaw=-18),
    # Do two things at a time:
    Action(forward=1, jumping=True),
    Action(forward=1, attacking=True),
    Action(jumping=True, attacking=True)
]
