import torch
from terminator import TerminatorClassicHypixel

from duels_training.benchmark_hypixel import main
from duels_training.classic_model import ClassicModel
from duels_training.classic_policy import sample_action
from duels_training.preprocessing import transform_raw_state


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


if __name__ == "__main__":
    model = ClassicModel()

    print(f"Connecting to terminator on localhost:6660")
    client = TerminatorClassicHypixel(capture_original_footage=True)
    client.connect(("localhost", 7000))

    main(
        game_name="classic",
        model=model,
        client=client,
        policy_state_class=PolicyState,
        filename="classic_benchmark_hypixel.csv",
        directory="classic_benchmark_hypixel"
    )
