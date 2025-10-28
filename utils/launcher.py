import logging
import os
import pathlib
import sys
import time
from typing import Callable, Literal

import ornamentalist
import submitit

log = logging.getLogger(__name__)


def prepare_submission(
    fn: Callable[[ornamentalist.ConfigDict], None],
    config: ornamentalist.ConfigDict,
) -> Callable:
    def submission():  # closure, equivalent to partial application
        return fn(config)

    # attach a checkpoint method so submitit knows that it should auto-requeue on timeout
    def checkpoint(*args, **kwargs):
        return submitit.helpers.DelayedSubmission(submission)

    setattr(submission, "checkpoint", checkpoint)
    return submission


@ornamentalist.configure(name="launcher")
def launch(
    fn: Callable[[ornamentalist.ConfigDict], None],
    configs: list[dict],
    nodes: int = ornamentalist.Configurable[1],
    gpus: int = ornamentalist.Configurable[1],
    cpus: int = ornamentalist.Configurable[12],
    ram: int = ornamentalist.Configurable[64],
    timeout: int = ornamentalist.Configurable[1440],
    output_dir: str = ornamentalist.Configurable["./outputs/"],
    cluster: Literal["debug", "local", "slurm"] = ornamentalist.Configurable["debug"],
    desc: str = ornamentalist.Configurable[""],
):
    """Thin wrapper that launches the main function with submitit.
    If multiple configs are provided, they will be launched as an array job/sweep.
    Note that all jobs in the array will be launched with the same SLURM parameters."""

    del desc  # desc is just some free text we can use to filter with wandb in the UI

    output_dir = os.path.join(output_dir, f"{time.time():.0f}")
    executor = submitit.AutoExecutor(folder=output_dir, cluster=cluster)
    executor.update_parameters(
        nodes=nodes,
        tasks_per_node=gpus,  # set ntasks = ngpus
        gpus_per_node=gpus,
        cpus_per_task=cpus,
        slurm_mem_per_gpu=f"{ram}G",
        timeout_min=timeout,
        stderr_to_stdout=True,
        slurm_signal_delay_s=120,
    )

    os.makedirs(output_dir, exist_ok=True)
    launch_cmd = f"{sys.executable} {' '.join(sys.argv)}"
    with open(os.path.join(output_dir, "launch_cmd.txt"), "w") as f:
        f.write(launch_cmd)

    snapshot_dir = os.path.join(output_dir, "snapshot")
    with submitit.helpers.RsyncSnapshot(pathlib.Path(snapshot_dir)):
        fns = [prepare_submission(fn, config=config) for config in configs]
        jobs = executor.submit_array(fns)
        log.info(f"Submitted {jobs=}")

        # if local or debug, wait for job to finish, otherwise exit script as soon as job is submitted
        if cluster == "local":
            log.info("Running job(s) locally using multiprocessing...")
            log.info(f"stdout and stderr for each process are logged to {output_dir}.")
            log.info("The job is in another process so you won't see anything here.")
            log.info("(But ctrl-c will still kill the job.)")
            _ = [j.results()[0] for j in jobs]

        elif cluster == "debug":
            log.info("Running job(s) in this process in debug mode...")
            log.info("pdb will open automatically on crash.")
            log.info("It's best to only use 1 GPU in this mode.")
            _ = [j.results()[0] for j in jobs]
