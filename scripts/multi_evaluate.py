"""Competition evaluation script for the multi-drone levels.

Mirrors `evaluate.py`, but uses `multi_sim.simulate` and the multi-level config
(`config.controller` is a list of drones, not a single drone).

Note:
    Please do not alter this script or ask the course supervisors first!
"""

import logging
from pathlib import Path

import numpy as np
from multi_sim import simulate

from lsy_drone_racing.utils import load_config

logger = logging.getLogger(__name__)


def main():
    """Run the multi-drone simulation N times and save the results as 'evaluation.csv'."""
    n_runs = 20
    config_file = "multi_level0.toml"
    config = load_config(Path(__file__).parents[1] / "config" / config_file)
    controllers = ",".join(controller["file"] for controller in config.controller)
    # ep_times: list (n_runs) of arrays (n_drones,); np.nan means that drone did not finish.
    ep_times = simulate(
        config=config_file, controllers=controllers, n_runs=n_runs, render=False
    )
    ep_times = np.asarray(ep_times)  # shape (n_runs, n_drones)
    n_drones = ep_times.shape[1]

    # By convention, the first controller in the config is the opponent, the second is the
    # drone we are benchmarking. A run counts as successful only if our drone finishes the
    # track completely AND finishes first (before the opponent).
    OPPONENT = 0
    OURS = 1

    for rank in range(n_drones):
        times = ep_times[:, rank]
        finished = ~np.isnan(times)
        n_failed = int((~finished).sum())
        finish_rate = finished.mean()
        if n_failed:
            logger.warning(
                f"Drone {rank}: {n_failed} run{'' if n_failed == 1 else 's'} did not finish out of {n_runs}!"
            )
        else:
            logger.info(f"Drone {rank}: all runs completed the track!")

        avg_time = np.mean(times[finished]) if finished.any() else float("nan")
        logger.info(f"Drone {rank}: Average Time (s): {avg_time}")
        logger.info(f"Drone {rank}: Finish Rate: {finish_rate * 100}%")

    our_times = ep_times[:, OURS]
    opp_times = ep_times[:, OPPONENT]
    our_finished = ~np.isnan(our_times)
    # Won = we finished, and either the opponent did not finish or we finished strictly earlier.
    won = our_finished & (np.isnan(opp_times) | (our_times < opp_times))
    n_won = int(won.sum())
    overall_success = won.mean()
    overall_avg = np.mean(our_times[won]) if won.any() else float("nan")

    logger.info(
        f"Our drone (index {OURS}) won {n_won} of {n_runs} runs "
        f"(finished the track first)."
    )
    logger.info(f"Success Rate (finished first): {overall_success * 100}%")
    logger.info(f"Average Winning Time (s): {overall_avg}")

    if overall_success < 0.5:
        logger.error("Our drone won fewer than 50% of runs! Aborting evaluation.")
        raise RuntimeError("Too many runs lost!")

    file = Path(__file__).parents[1] / "evaluation.csv"
    with open(file, "w") as f:
        f.write(f"{overall_avg},{overall_success},")
    logger.info(f"Results saved in {file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
