from terminator import TerminatorSumoHypixel

from duels_training.benchmark_hypixel import main
from duels_training.sumo_demo_utils import PolicyState
from duels_training.sumo_model import SumoModel

if __name__ == "__main__":
    model = SumoModel()

    print(f"Connecting to terminator on localhost:6660")
    client = TerminatorSumoHypixel(capture_original_footage=True)
    client.connect(("localhost", 6660))

    main(
        game_name="sumo",
        model=model,
        client=client,
        policy_state_class=PolicyState,
        filename="sumo_benchmark_hypixel.csv",
        directory="sumo_benchmark_hypixel"
    )
