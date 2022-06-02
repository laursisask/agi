import argparse
import os
import random
import readline
from threading import Thread

import cv2
import torch
from terminator import TerminatorBridge

from duels_training.bridge_maps import MAPS
from duels_training.bridge_model import BridgeModel
from duels_training.bridge_utils import PolicyState


class Recorder:
    def __init__(self, directory, episode_index):
        file = os.path.join(directory, f"{episode_index}.mp4")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(file, fourcc, 20, (336, 336), True)

    def record_image(self, tensor):
        self.video_writer.write(cv2.cvtColor(cv2.flip(tensor.numpy(), 0), cv2.COLOR_RGB2BGR))

    def close(self):
        self.video_writer.release()


def play_episode(model, device, client, session, map_name, video_counter):
    recorder = Recorder("bridge_videos", video_counter)

    if map_name is None:
        map_name = random.choice(MAPS)

    observation = client.reset(session=session, map_name=map_name, spawn_distance=1)
    policy_state = PolicyState(model=model, device=device, observation=observation)

    recorder.record_image(observation.original_footage)

    done = False
    while not done:
        action = policy_state.sample_action()
        next_observation, _, done, metadata = client.step(action)

        policy_state.update(next_observation)

        recorder.record_image(next_observation.original_footage)


def play_agent_against_itself(model, device, num_episodes, map_name):
    print(f"Connecting to terminator on localhost:6660")
    client1 = TerminatorBridge(capture_original_footage=True)
    client1.connect(("localhost", 6660))

    print(f"Connecting to terminator on localhost:6661")
    client2 = TerminatorBridge(capture_original_footage=True)
    client2.connect(("localhost", 6661))

    video_counter = 0

    for i in range(num_episodes):
        session = client1.create_session()

        t1 = Thread(target=play_episode, args=(model, device, client1, session, map_name, video_counter), daemon=True)
        t2 = Thread(target=play_episode, args=(model, device, client2, session, map_name, video_counter + 1),
                    daemon=True)

        video_counter += 2

        t1.start()
        t2.start()

        t1.join()
        t2.join()


def play_against_player(model, device, map_name):
    print(f"Connecting to terminator on localhost:6660")
    client = TerminatorBridge(capture_original_footage=True)
    client.connect(("localhost", 6660))

    readline.set_auto_history(True)

    video_counter = 0

    while session := input("Enter session name: "):
        play_episode(model, device, client, session, map_name, video_counter)
        video_counter += 1


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--interactive", type=bool, default=False)
    parser.add_argument("--map", choices=MAPS)

    args = parser.parse_args()

    model = BridgeModel()

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
