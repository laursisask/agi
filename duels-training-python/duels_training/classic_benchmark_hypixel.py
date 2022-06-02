from terminator import TerminatorClassicHypixel

from duels_training.benchmark_hypixel import main
from duels_training.classic_model import ClassicModel
from duels_training.classic_utils import PolicyState

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
