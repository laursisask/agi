import math
import os

import pytest

from duels_training.incremental_stats_calculator import IncrementalStatsCalculator


def test_std(tmp_path):
    calculator = IncrementalStatsCalculator()

    calculator.append(3)
    calculator.append(7)
    assert calculator.std() == pytest.approx(math.sqrt(8))

    calculator.append(5)
    assert calculator.std() == pytest.approx(2)

    calculator.append(-10.0)
    assert calculator.std() == pytest.approx(7.676, abs=1e-3)

    file_path = os.path.join(tmp_path, "incremental_stats.json")
    with open(file_path, "w") as file:
        calculator.dump(file)

    new_calculator = IncrementalStatsCalculator()
    with open(file_path, "r") as file:
        new_calculator.load(file)

    assert new_calculator.std() == pytest.approx(7.676, abs=1e-3)


def test_std_zero():
    calculator = IncrementalStatsCalculator()

    calculator.append(100)
    calculator.append(100)
    calculator.append(100)
    calculator.append(100)
    calculator.append(100)

    assert calculator.std() == 0
