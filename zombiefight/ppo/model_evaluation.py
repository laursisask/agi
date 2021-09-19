import argparse
import time

import cv2
import torch
from terminator import Terminator

from ppo import action_utils
from ppo.distributions import ModelOutputDistribution
from ppo.model import PPOModel
from processing import create_observation
from stacked_state_constructor import StackedStateConstructor


def main():
    num_of_episodes = 100
    stacked_frame_count = 4

    client = Terminator()
    client.connect(("localhost", 6660))
    print(f"Connected to MC client on port 6660")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = PPOModel()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.to(device)
    model.eval()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    original_video_writer = cv2.VideoWriter("ppo/original-video.mp4", fourcc, 20, (336, 336), True)
    agent_video_writer = cv2.VideoWriter("ppo/agent-video.mp4", fourcc, 20, (84, 84), False)

    @torch.inference_mode()
    def pick_action(state):
        images, others = state
        images = images.to(device)
        others = others.to(device)

        output = model(images=images.unsqueeze(0), others=others.unsqueeze(0))

        dist = ModelOutputDistribution(output)

        action = action_utils.to_python_types(dist.sample())

        return action

    episode_rewards = []

    for i in range(num_of_episodes):
        done = False
        episode_reward = 0
        episode_start = time.time()
        step_count = 0

        initial_raw_observation = client.reset()
        initial_agent_observation = create_observation(initial_raw_observation)
        state_constructor = StackedStateConstructor(stacked_frame_count)
        state_constructor.append(initial_agent_observation)

        original_video_writer.write(cv2.cvtColor(initial_raw_observation.camera.numpy(), cv2.COLOR_RGB2BGR))
        agent_video_writer.write(
            (initial_agent_observation.image * 256 + 128).to(dtype=torch.uint8).reshape([84, 84, 1]).numpy()
        )

        while not done:
            action = pick_action(state_constructor.current())
            next_raw_observation, reward, done = client.step(action)

            episode_reward += reward
            step_count += 1
            next_observation = create_observation(next_raw_observation)
            state_constructor.append(next_observation)

            original_video_writer.write(cv2.cvtColor(next_raw_observation.camera.numpy(), cv2.COLOR_RGB2BGR))
            agent_video_writer.write(
                (next_observation.image * 256 + 128).to(dtype=torch.uint8).reshape([84, 84, 1]).numpy()
            )

        episode_rewards.append(episode_reward)

        episode_duration = time.time() - episode_start
        tps = step_count / episode_duration
        print(f"Episode finished with reward {episode_reward:1f}, took {episode_duration:.1f} seconds, TPS {tps:.1f}")

    original_video_writer.release()
    agent_video_writer.release()

    print(f"Average reward per episode: {sum(episode_rewards) / len(episode_rewards)}")


if __name__ == "__main__":
    main()
