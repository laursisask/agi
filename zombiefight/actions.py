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
    Action(looking=1),
    Action(looking=-1),
    Action(turning=1),
    Action(turning=-1),
    # Do two things at a time:
    Action(forward=1, jumping=True),
    Action(forward=1, attacking=True),
    Action(jumping=True, attacking=True)
]
