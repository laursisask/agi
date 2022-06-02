import random

from terminator import TerminatorSumo

from duels_training.random_baseline import run_baseline
from duels_training.sumo_demo_utils import PolicyState
from duels_training.sumo_maps import MAPS
from duels_training.sumo_model import SumoModel

if __name__ == "__main__":
    def get_extra_reset_args():
        return {
            "map_name": random.choice(MAPS),
            "random_teleport": False,
            "randomization_factor": 1
        }


    run_baseline(SumoModel, TerminatorSumo, PolicyState, get_extra_reset_args)
