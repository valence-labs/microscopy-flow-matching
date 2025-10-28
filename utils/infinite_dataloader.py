import itertools
import logging
from typing import Generator, Iterator

from torch.utils.data import DataLoader, Sampler

log = logging.getLogger(__name__)


def infinite_dataloader(
    dataloader: DataLoader,
    start_step: int = 0,
    fast_resume: bool = True,
) -> Generator:
    """Wraps a DataLoader to allow you to yield batches indefinitely.
    Internally handles calling sampler.set_epoch() to ensure correct shuffling in DDP.

    If start_step is provided, the dataloader will resume from the given step by
    skipping the appropriate number of batches (i.e. start_step % len(dataloader)).

    Internally, skipping these initial steps requires loading and discarding
    each batch, which can be expensive, especially for large datasets.
    If fast_resume is True, we will not do this. Instead, we will resume
    training from the next epoch. This trades off exact reproducibility for
    speed. If your dataset is large and/or shuffled, this setting is recommended.

    Usage:
    ```python
    global_step = 0  # or whatever step you want to resume from
    train_loader = infinite_dataloader(train_loader, start_step=global_step)
    for _, batch in zip(range(global_step, max_steps), train_loader):
        ...
    ```
    """

    sampler: Sampler | None = getattr(dataloader, "sampler", None)

    try:
        steps_per_epoch = len(dataloader)
        if steps_per_epoch <= 0:
            raise ValueError("DataLoader length must be positive.")
    except TypeError as e:
        raise ValueError("DataLoader must support len() for infinite_dataloader") from e

    if start_step < 0:
        raise ValueError("start_step cannot be negative")

    current_epoch = start_step // steps_per_epoch

    if not fast_resume:
        steps_to_skip = start_step % steps_per_epoch
    else:
        steps_to_skip = 0
        if start_step > 0:
            current_epoch += 1  # resume from start of next epoch

    while True:  # loop indefinitely over epochs
        if isinstance(sampler, Sampler) and hasattr(sampler, "set_epoch"):
            log.info(f"Setting sampler epoch to {current_epoch}")
            sampler.set_epoch(current_epoch)  # type: ignore
        else:
            log.warning("Sampler does not support set_epoch or is missing.")
            log.warning("This may affect how data is shuffled in DDP settings.")

        epoch_iterator: Iterator = iter(dataloader)

        if steps_to_skip > 0:
            log.info(f"Resuming training from step {start_step}.")
            epoch_iterator = itertools.islice(epoch_iterator, steps_to_skip, None)
            steps_to_skip = 0  # reset to avoid skipping steps again

        yield from epoch_iterator
        current_epoch += 1
