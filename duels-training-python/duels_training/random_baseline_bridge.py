import random

from terminator import TerminatorBridge

from duels_training.bridge_maps import MAPS
from duels_training.bridge_model import BridgeModel
from duels_training.bridge_utils import PolicyState
from duels_training.random_baseline import run_baseline


# partial results:
# 3 wins
# 150 draws

def get_extra_reset_args():
    return {
        "map_name": random.choice(MAPS),
        "spawn_distance": 1
    }


if __name__ == "__main__":
    run_baseline(BridgeModel, TerminatorBridge, PolicyState, get_extra_reset_args)
