# Experiment 1. Fighting Zombies using Deep Q-Learning

In the first experiment, the agent has to fight a zombie in a static world.

This is a very simple scenario that could easily be solved by a handwritten
algorithm. The main purpose was to see if it could also be solved by a
reinforcement learning algorithm.

In this directory, you will find
- the source code for training and evaluating an agent/model
- weights for the models after 500K iterations
- video examples of the agent playing (`original-video.mp4` and `agent-video.mp4`)

Unfortunately, there is also another part (`terminator`) required to run this
experiment which is not available publicly.  This is a modified version of
Minecraft which implements the environment and handles the communication
between Minecraft process and Python. The reason it is not available is because
it contains the code for Minecraft and uploading that is illegal.

## Environment

The play area is a 16x16 rectangle which is surrounded by a fence.

When an episode begins, the player is spawned in one specific corner of the
play area and the zombie's spawn location is chosen randomly. The goal is to
kill the zombie. The player gets a reward of 1/20 * damage done every time they
damage the zombie and an additional 10 reward when the zombie dies.

The episode ends when the agent or zombie dies, or a time limit of 1200 ticks
(around 60 seconds in wall time).

At every time step, the player can take one of the following actions:
- Do nothing
- Go forward (press W on keyboard)
- Go backwards (press S on keyboard)
- Go left (press A on keyboard)
- Go right (press D on keyboard)
- Jump (press space)
- Attack (left-click)
- Look 9 degrees higher
- Look 9 degrees lower
- Turn 18 degrees clockwise
- Turn 18 degrees counter-clockwise
- Go forward and jump
- Go forward and attack
- Jump and attack

Minecraft version 1.8 was used.

## Algorithms

I used double-learning, i.e. had two independent estimates for action
values and used both of them to estimate the value of the next state.

Target networks were used as well, i.e. during training we estimated the next
state's value with neural networks whose parameters were "lagging behind" the
optimized neural networks.

A replay buffer with size of 50000 was used to sample the batches for model
optimization.

## Data and Model

As input, the model got the downscaled grayscale 84x84 image of the last four
frames.  We ran Minecraft in 336x336 and downscaled the image to 84x84.

Additionally, I gave the following features to the model directly (as numbers):
- zombie's relative location to the player (in all x, y, z dimensions)
- player's pitch (vertical rotation, how is the player looking up/down)
- player's yaw (horizontal rotation, how is the player looking left/right)
- player's health
- zombie's health

I used a model almost identical to the one introduced in DeepMind's Atari
paper. The only difference was that we concatted the output from convolutional
layers with the additional features (the ones in previous paragraph).

## Results

I trained the model for 500K iterations. At 500K iterations, the model beat
the zombie in 99.5% of cases.

# Experiment 2. Fighting Zombies with PPO

The environment, data and model were very similar to the Q-learning one.

One difference was that instead of having fixed actions, a choice was made in
every dimension - the model predicted a value for all of the actions
independently. The mouse related actions were continuous and the mean and
standard deviation of the actions was predicted.
