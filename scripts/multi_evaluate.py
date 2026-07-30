"""Competition evaluation script for the multi-drone levels.

Mirrors `evaluate.py`, but uses `multi_sim.simulate` and the multi-level config
(`config.controller` is a list of drones, not a single drone).

Note:
    Please do not alter this script or ask the course supervisors first!
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from multi_sim import simulate

from lsy_drone_racing.utils import load_config

logger = logging.getLogger(__name__)

def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def main():
    """Run the multi-drone simulation N times and save the results as 'evaluation.csv'."""
    n_runs = 40
    n_fit_memory = 5
    assert n_runs % n_fit_memory == 0
    n_loops = int(n_runs/n_fit_memory)

    config_file = "deploy_multi.toml"
    config = load_config(Path(__file__).parents[1] / "config" / config_file)
    controllers = ",".join(controller["file"] for controller in config.controller)
    ep_times = []

    
    for _ in range(n_loops):
        # ep_times: list (n_runs) of arrays (n_drones,); np.nan means that drone did not finish.
        ep_time = simulate(
            config=config_file, controllers=controllers, n_runs=n_fit_memory, render=False
        )
        ep_times.extend(ep_time)

    ep_times = np.asarray(ep_times)  # shape (n_runs, n_drones)
    
    '''

    ep_times = np.array([[  np.nan,   np.nan],
            [10.5 ,   np.nan],
            [  np.nan,   np.nan],
            [5.44,   6.55],
            [6.88,   5.77],
            [6.34,   7.44],
            [  np.nan,   np.nan]], dtype=np.float32)
    '''

    finished_1 = ~np.isnan(ep_times[:, 0])
    finished_2 = ~np.isnan(ep_times[:, 1])

    both_finished = int(np.sum(finished_1 & finished_2))
    both_finished_percent = 100.0 * both_finished / n_runs

    none_finished = int(np.sum(~finished_1 & ~finished_2))
    only_1_finished = int(np.sum(finished_1 & ~finished_2))
    only_2_finished = int(np.sum(~finished_1 & finished_2))

    # Only count overtakes when both drones finished.
    overtakes = int(np.sum(finished_1 & finished_2 & (ep_times[:, 1] < ep_times[:, 0])))
    percent_overtake = (
        100.0 * overtakes / both_finished if both_finished > 0 else 0.0
    )

    logger.info("========== Race Statistics ==========")
    logger.info(f"Both finished:          {both_finished}/{n_runs} ({both_finished_percent:.1f}%)")
    logger.info(f"Only drone 1 finished:  {only_1_finished}")
    logger.info(f"Only drone 2 finished:  {only_2_finished}")
    logger.info(f"None finished:          {none_finished}")
    logger.info(f"Overtakes:              {overtakes}/{both_finished} ({percent_overtake:.1f}%)")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    results = {
        "timestamp": timestamp,
        "config_file": config_file,
        "controllers": [controller["file"] for controller in config.controller],
        "simulation": {
            "n_runs": n_runs,
        },
        "episode_times": ep_times.tolist(),
        "metrics": {
            "both_finished": both_finished,
            "both_finished_percent": both_finished_percent,

            "none_finished": none_finished,

            "only_drone_1_finished": only_1_finished,
            "only_drone_2_finished": only_2_finished,

            "overtakes": overtakes,
            "overtakes_percent": percent_overtake,
        },
        "config": config.to_dict(),
    }

    result_dir = Path(__file__).parents[1] / "evaluation_results/multi_sim"
    result_dir.mkdir(parents=True, exist_ok=True)

    result_file = result_dir / f"{timestamp}.json"

    with open(result_file, "w") as f:
        json.dump(results, f, indent=4, default=json_default)

    logger.info(f"Results saved to {result_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
