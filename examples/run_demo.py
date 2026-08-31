"""Run the checked-in OrchidRec example from any working directory."""

import json
from pathlib import Path

from orchidrec import load_config, run_experiment

if __name__ == "__main__":
    config_path = Path(__file__).with_name("config.json")
    result = run_experiment(load_config(config_path))
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
