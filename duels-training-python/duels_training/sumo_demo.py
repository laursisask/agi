import argparse
from threading import Thread

import torch
from terminator import Terminator

from duels_training.sumo_model import SumoModel
from duels_training.sumo_policy import sample_action
from duels_training.sumo_preprocessing import transform_raw_state

# mark the import as used
# Torch looks for the imported class "SumoModel" inside the
# current module when it loads the model from disk
__x = SumoModel()


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


def play_episode(model, device, client, session):
    observation = client.reset(session)
    policy_state = PolicyState(model=model, device=device, observation=observation)

    done = False
    while not done:
        action = policy_state.sample_action()
        next_observation, _, done, _ = client.step(action)

        policy_state.update(next_observation)


def play_agent_against_itself(model, device, num_episodes):
    print(f"Connecting to terminator on localhost:6660")
    client1 = Terminator()
    client1.connect(("localhost", 6660))

    print(f"Connecting to terminator on localhost:6661")
    client2 = Terminator()
    client2.connect(("localhost", 6661))

    for i in range(num_episodes):
        session = client1.create_session()

        t1 = Thread(target=play_episode, args=(model, device, client1, session), daemon=True)
        t2 = Thread(target=play_episode, args=(model, device, client2, session), daemon=True)

        t1.start()
        t2.start()

        t1.join()
        t2.join()


def play_against_player(model, device):
    print(f"Connecting to terminator on localhost:6660")
    client = Terminator()
    client.connect(("localhost", 6660))

    while True:
        session = input("Enter session name: ")
        play_episode(model, device, client, session)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--interactive", type=bool, default=False)

    args = parser.parse_args()

    with open(args.model, "rb") as f:
        model = torch.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    model.to(device)

    if args.interactive:
        play_against_player(model, device)
    else:
        play_agent_against_itself(model, device, args.num_episodes)


if __name__ == "__main__":
    main()

# Top iterations in my opinion:
# 1. 2179
# 2. 2200
# 3. 1545
# 4. 1537
