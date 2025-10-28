from typing import Any, Callable

import torch
from tabulate import tabulate
from torch.utils.flop_counter import FlopCounterMode

BYTES_PER_GIB = 1024**3


def get_model_summary(model: torch.nn.Module, fwd_fn: Callable[[], Any]) -> str:
    """Get a summary of the model's parameters, FLOPs, and other metrics.
    `fwd_fn` is the function whose flops are counted. It should take no arguments,
    and its return value is ignored.

    You typically want to do something like this:
    ```python
    x = torch.randn(1, 6, 128, 128) # 6-channel 128x128 input image
    y = torch.randn(1, 1024)        # 1024-dim input vector
    fwd_fn = lambda: model(x) # e.g. a model which only takes x as input
    fwd_fn = lambda: complex_model(x, y) # e.g. a model taking x and y as input
    ```
    Notes:
    1. FLOPs are floating point operations, a measure of computational cost.
    Not to be confused with FLOPs/s, which are a measure of throughput.
    FLOPs/s are hardware-dependent, and are not reported here.

    2. Backward pass FLOPs can be approximated as ~2x forward pass FLOPs.

    3. You usually want to use a batch size of 1 when computing FLOPs.
    It's common practice to report forward pass FLOPs per single sample.
    You can approximate total FLOPs used for training by:
    FLOPs per sample * effective batch size * number of training steps * 3.

    4. This function reports both FLOPs (floating point operations) and
    MACs (multiply-accumulate operations). MACs are simply approximated as
    FLOPs / 2. If comparing to other papers, note that most ML papers
    which claim to report FLOPs are actually reporting MACs.

    E.g. try running the following:
    ```python
    from torchvision.models.convnext import convnext_large
    model = convnext_large()
    fwd_fn = lambda: model(torch.randn(1, 3, 224, 224))
    print(get_model_summary(model, fwd_fn))
    ```

    Compare the output to the results reported in the ConvNeXt paper [1].
    You'll find that our 'MACs' are equal to the 'FLOPs' reported in the paper.
    [1] https://arxiv.org/abs/2201.03545
    """
    istrain = model.training
    model.eval()  # set to eval mode to avoid messing up batch stats
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        fwd_fn()

    # restore model to original mode
    if istrain:
        model.train()

    total_flops = flop_counter.get_total_flops()
    total_macs = total_flops / 2

    param_dtype = next(model.parameters()).dtype
    param_size = next(model.parameters()).element_size()  # in bytes

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    summary = tabulate(
        [
            ["Model", model.__class__.__name__],
            ["Forward pass FLOPs", f"{total_flops / 1e9:,.2f} B"],
            ["Forward pass MACs", f"{total_macs / 1e9:,.2f} B"],
            ["Trainable params", f"{trainable_params / 1e6:,.2f} M"],
            ["Total params", f"{total_params / 1e6:,.2f} M"],
            ["Param dtype", f"{param_dtype}"],
            ["Param size", f"{(param_size * total_params) / BYTES_PER_GIB:,.2f} GiB"],
        ],
        tablefmt="rst",
    )
    return summary


if __name__ == "__main__":
    """If run as a script, print this example to compare with the ConvNeXt paper."""
    from torchvision.models.convnext import convnext_large

    model = convnext_large()
    print(get_model_summary(model, lambda: model(torch.randn(1, 3, 224, 224))))
