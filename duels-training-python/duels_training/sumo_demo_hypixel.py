import argparse
import math
import readline

import torch
from terminator import TerminatorSumoHypixel

from duels_training.sumo_demo_utils import PolicyState
from duels_training.sumo_model import SumoModel


def play_episode(model, device, client, opponent):
    observation = client.reset(opponent)
    policy_state = PolicyState(model=model, device=device, observation=observation)

    metadata = {}
    done = False
    while not done:
        action = policy_state.sample_action()
        next_observation, _, done, metadata = client.step(action)

        policy_state.update(next_observation)

    return math.isclose(metadata.get("win", 0), 1)


def play_against_player(model, device):
    print(f"Connecting to terminator on localhost:6660")
    client = TerminatorSumoHypixel()
    client.connect(("localhost", 6660))

    readline.set_auto_history(True)

    while opponent := input("Enter opponent's name: "):
        did_win = play_episode(model, device, client, opponent)

        if did_win:
            print("Agent won")
        else:
            print("Agent lost")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)

    args = parser.parse_args()

    model = SumoModel()

    with open(args.model, "rb") as f:
        model.load_state_dict(torch.load(f, map_location="cpu"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    model.to(device)
    model.eval()

    play_against_player(model, device)


if __name__ == "__main__":
    main()
