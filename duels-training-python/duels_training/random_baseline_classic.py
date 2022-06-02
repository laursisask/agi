import random

from terminator import TerminatorClassic

from duels_training.classic_maps import MAPS
from duels_training.classic_model import ClassicModel
from duels_training.classic_utils import PolicyState
from duels_training.random_baseline import run_baseline


def get_extra_reset_args():
    return {
        "map_name": random.choice(MAPS),
        "random_teleport": False,
        "spawn_distance": 1
    }


if __name__ == "__main__":
    run_baseline(ClassicModel, TerminatorClassic, PolicyState, get_extra_reset_args)
