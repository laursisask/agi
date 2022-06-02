import random
from argparse import ArgumentParser
from threading import Thread

import torch
from terminator import Action


@torch.no_grad()
def play_according_to_model(device, model, policy_state_class, client, session, extra_reset_args):
    observation = client.reset(session=session, **extra_reset_args)
    policy_state = policy_state_class(model=model, device=device, observation=observation)

    outcome = "loss"

    done = False
    while not done:
        action = policy_state.sample_action()
        next_observation, _, done, metadata = client.step(action)
        policy_state.update(next_observation)

        if "win" in metadata:
            outcome = "win"
        elif "timeout" in metadata:
            outcome = "timeout"

    return outcome


def sample_random_action():
    return Action(
        forward=random.choice([1, 0, -1]),
        left=random.choice([1, 0, -1]),
        jumping=random.choice([True, False]),
        attacking=random.choice([True, False]),
        sprinting=random.choice([True, False]),
        using=random.choice([True, False]),
        sneaking=random.choice([True, False]),
        delta_yaw=random.uniform(-9, 9),
        delta_pitch=random.uniform(-9, 9),
        inventory_slot=random.randint(0, 8)
    )


def play_randomly(client, session, extra_reset_args):
    client.reset(session=session, **extra_reset_args)

    done = False
    while not done:
        _, _, done, _ = client.step(sample_random_action())


def play_episode(device, model, policy_state_class, get_extra_reset_args, client1, client2):
    session = client1.create_session()

    extra_reset_args = get_extra_reset_args()

    random_agent_thread = Thread(target=play_randomly, args=(client2, session, extra_reset_args), daemon=True)
    random_agent_thread.start()

    outcome = play_according_to_model(device, model, policy_state_class, client1, session, extra_reset_args)
    random_agent_thread.join()
    return outcome


def run_baseline(model_class, terminator_class, policy_state_class, get_extra_reset_args):
    parser = ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-episodes", default=100, type=int)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device} device")

    model = model_class()
    print(f"Loading model from {args.model}")
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.to(device)
    model.eval()

    print(f"Connecting to terminator on localhost:6660")
    client1 = terminator_class()
    client1.connect(("localhost", 6660))

    print(f"Connecting to terminator on localhost:6661")
    client2 = terminator_class()
    client2.connect(("localhost", 6661))

    wins = 0
    timeouts = 0
    losses = 0

    for i in range(args.num_episodes):
        print(f"Starting episode {i}")

        outcome = play_episode(device, model, policy_state_class, get_extra_reset_args, client1, client2)

        print(f"Episode finished, outcome: {outcome}")

        if outcome == "win":
            wins += 1
        elif outcome == "timeout":
            timeouts += 1
        else:
            losses += 1

    print("Final results:")
    print(f"Wins: {wins}, {wins / args.num_episodes * 100:.1f}%")
    print(f"Timeouts: {timeouts}, {timeouts / args.num_episodes * 100:.1f}%")
    print(f"Losses: {losses}, {losses / args.num_episodes * 100:.1f}%")
