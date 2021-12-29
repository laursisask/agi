import argparse
import random
from threading import Thread

import torch
from terminator import TerminatorSumo

from duels_training.sumo_maps import MAPS
from duels_training.sumo_model import SumoModel
from duels_training.sumo_policy import sample_action
from duels_training.sumo_preprocessing import transform_raw_state


class PolicyState:
    def __init__(self, model, device, observation):
        self.model = model
        self.device = device
        self.last_state = None
        self.policy_output = None

        self.update(observation)

    @torch.no_grad()
    def update(self, observation):
        policy_output, _, self.last_state = self.model(
            transform_raw_state(observation).unsqueeze(0).unsqueeze(0).to(self.device),
            initial_state=self.last_state,
            compute_value_estimates=False
        )

        self.policy_output = policy_output.to(torch.device("cpu")).squeeze(1)

    @torch.no_grad()
    def sample_action(self):
        return sample_action(self.policy_output)


def play_episode(model, device, client, session, map_name):
    if map_name is None:
        map_name = random.choice(MAPS)

    observation = client.reset(session=session, randomization_factor=1, map_name=map_name)
    policy_state = PolicyState(model=model, device=device, observation=observation)

    done = False
    while not done:
        action = policy_state.sample_action()
        next_observation, _, done, _ = client.step(action)

        policy_state.update(next_observation)


def play_agent_against_itself(model, device, num_episodes, map_name):
    print(f"Connecting to terminator on localhost:6660")
    client1 = TerminatorSumo()
    client1.connect(("localhost", 6660))

    print(f"Connecting to terminator on localhost:6661")
    client2 = TerminatorSumo()
    client2.connect(("localhost", 6661))

    for i in range(num_episodes):
        session = client1.create_session()

        t1 = Thread(target=play_episode, args=(model, device, client1, session, map_name), daemon=True)
        t2 = Thread(target=play_episode, args=(model, device, client2, session, map_name), daemon=True)

        t1.start()
        t2.start()

        t1.join()
        t2.join()


def play_against_player(model, device, map_name):
    print(f"Connecting to terminator on localhost:6660")
    client = TerminatorSumo()
    client.connect(("localhost", 6660))

    while True:
        session = input("Enter session name: ")
        play_episode(model, device, client, session, map_name)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--interactive", type=bool, default=False)
    parser.add_argument("--map", choices=MAPS)

    args = parser.parse_args()

    model = SumoModel()

    with open(args.model, "rb") as f:
        model.load_state_dict(torch.load(f, map_location="cpu"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    model.to(device)
    model.eval()

    if args.interactive:
        play_against_player(model, device, args.map)
    else:
        play_agent_against_itself(model, device, args.num_episodes, args.map)


if __name__ == "__main__":
    main()

# Top iterations in my opinion:
# 1. 2179
# 2. 2200
# 3. 1545
# 4. 1537
