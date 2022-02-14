import argparse
import csv
import os.path
import time

import cv2
import requests
import torch
from requests.adapters import HTTPAdapter
from terminator import TerminatorSumoHypixel
from urllib3.util import Retry

from duels_training.sumo_demo_utils import PolicyState
from duels_training.sumo_model import SumoModel

http = requests.Session()
http.mount("https://", HTTPAdapter(max_retries=Retry(total=7, backoff_factor=0.1, status_forcelist={429})))


class Recorder:
    def __init__(self, episode_index):
        file = os.path.join("sumo_benchmark_hypixel", f"{episode_index}.mp4")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(file, fourcc, 20, (336, 336), True)

    def record_image(self, tensor):
        self.video_writer.write(cv2.cvtColor(cv2.flip(tensor.numpy(), 0), cv2.COLOR_RGB2BGR))

    def close(self):
        self.video_writer.release()


def get_player_uuid(username):
    response = http.get(f"https://api.mojang.com/users/profiles/minecraft/{username}")
    response.raise_for_status()

    # When player does not exist, 204 is returned
    if response.status_code == 200:
        return response.json()["id"]
    else:
        return None


def dig(dictionary, keys, default=None):
    value = dictionary

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def get_player_stats(username, api_key):
    player_uuid = get_player_uuid(username)

    if player_uuid is None:
        return None

    response = http.get("https://api.hypixel.net/player", params={"uuid": player_uuid},
                        headers={"API-Key": api_key})
    response.raise_for_status()

    retrieved_keys = ["current_winstreak", "rounds_played", "sumo_duel_wins", "sumo_duel_rounds_played"]

    all_stats = response.json()

    # Not all keys are always present. For example, when player has not
    # won a single game in sumo.
    return {key: dig(all_stats, ["player", "stats", "Duels", key], 0) for key in retrieved_keys}


def play_episode(model, device, client, episode_index):
    observation = client.reset(None)
    policy_state = PolicyState(model=model, device=device, observation=observation)

    recorder = Recorder(episode_index)
    recorder.record_image(observation.original_footage)

    start_time = time.time()
    map_name = None
    opponent_name = None
    metadata = {}
    done = False
    while not done:
        action = policy_state.sample_action()
        next_observation, _, done, metadata = client.step(action)

        if "opponent_name" in metadata:
            opponent_name = metadata["opponent_name"]
            print(f"Playing against {opponent_name}")

        if "map_name" in metadata:
            map_name = metadata["map_name"]
            print(f"Map: {map_name}")

        policy_state.update(next_observation)

        recorder.record_image(next_observation.original_footage)

    result = metadata["result"]

    end_time = time.time()
    duration = end_time - start_time

    recorder.close()

    return result, map_name, opponent_name, duration


def evaluate(model, device, num_episodes, api_key):
    filename = "sumo_benchmark_hypixel.csv"
    print(f"Writing results to {filename}")
    file = open(filename, "a")
    csv_writer = csv.writer(file, dialect=csv.unix_dialect)

    first_episode = 0
    wins = 0

    if file.tell() == 0:
        csv_writer.writerow(["episode", "result", "duration", "map_name", "opponent_name",
                             "opponent_current_winstreak", "opponent_rounds_played", "opponent_sumo_duel_wins",
                             "opponent_sumo_duel_rounds_played"])
    else:
        print("Benchmark file already exists")
        with open(filename, "r") as file_for_read:
            csv_reader = csv.reader(file_for_read, dialect=csv.unix_dialect)
            lines = list(csv_reader)

            if len(lines) > 1:
                first_episode = int(lines[-1][0]) + 1
                print(f"Continuing from episode {first_episode}")

            wins = sum(1 if line == "True" else 0 for line in lines[1:])

    print(f"Connecting to terminator on localhost:6660")
    client = TerminatorSumoHypixel(capture_original_footage=True)
    client.connect(("localhost", 6660))

    for i in range(first_episode, num_episodes):
        print(f"Starting episode {i}")
        result, map_name, opponent_name, duration = play_episode(model, device, client, i)

        if result == "victory":
            wins += 1
            print("Agent won the game")
        elif result == "defeat":
            print("Agent lost the game")
        elif result == "draw":
            print("Game ended in a draw")
        else:
            raise ValueError(f"Unknown result {result}")

        stats = get_player_stats(opponent_name, api_key)
        if stats is None:
            print(f"Could not find player {opponent_name}")
            i -= 1
        else:
            print(f"{opponent_name}'s stats: general duels winstreak {stats['current_winstreak']}, "
                  f"duels rounds played {stats['rounds_played']}, sumo rounds played {stats['sumo_duel_rounds_played']}, "
                  f"sumo wins {stats['sumo_duel_wins']}")

            csv_writer.writerow([i, result, duration, map_name, opponent_name, stats['current_winstreak'],
                                 stats['rounds_played'], stats['sumo_duel_wins'], stats['sumo_duel_rounds_played']])
            file.flush()
        print("----------")

    print("Evaluation finished")
    print(f"Win rate: {wins / num_episodes:.3f}")

    file.close()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-episodes", default=100, type=int)
    parser.add_argument("--api-key", required=True)

    args = parser.parse_args()

    model = SumoModel()

    with open(args.model, "rb") as f:
        model.load_state_dict(torch.load(f, map_location="cpu"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    model.to(device)
    model.eval()

    evaluate(model, device, args.num_episodes, args.api_key)


if __name__ == "__main__":
    main()
