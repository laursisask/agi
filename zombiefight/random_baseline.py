import random

from terminator import Terminator
from actions import ACTIONS


def main():
    num_of_episodes = 100

    client = Terminator()
    client.connect(("localhost", 6660))
    print(f"Connected to MC client on port 6660")

    episode_rewards = []

    for i in range(num_of_episodes):
        done = False
        episode_reward = 0
        client.reset()

        while not done:
            _, reward, done = client.step(random.choice(ACTIONS))

            episode_reward += reward

        episode_rewards.append(episode_reward)

    print(f"Average reward per episode: {sum(episode_rewards) / len(episode_rewards)}")


if __name__ == "__main__":
    main()
