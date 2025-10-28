import dataclasses
import logging
import os
import re
import tempfile
import time
from typing import Literal

import numpy as np
import torch
import torchdiffeq
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
from tqdm import tqdm

from bbbc021 import decode
from model import DiTWrapper
from utils.distributed import Distributed, rank_zero
from utils.ema import KarrasEMA
from utils.evaluation import FeatureExtractor, compute_fd, compute_kd
from utils.infinite_dataloader import infinite_dataloader
from utils.profiler import get_profiler

_log = logging.getLogger(__name__)


@rank_zero()
def log(*, step: int, msg: str | None = None, data: dict | None = None):
    if msg is not None:
        _log.info(f"[{step=}] {msg}")
    if data is not None:
        wandb.log(data, step=step)


@dataclasses.dataclass
class TrainState:
    ddp: DDP  # Distributed wrapper around DiT model
    ema: KarrasEMA  # EMA wrapper around DiT model
    opt: torch.optim.Optimizer
    global_step: int

    def _to_dict(self) -> dict:
        return {
            "global_step": self.global_step,
            "ddp": self.ddp.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
        }

    @rank_zero()
    def save_ckpt(self, path: str) -> None:
        dir = os.path.dirname(path)
        with tempfile.NamedTemporaryFile(delete=False, dir=dir, suffix=".tmp") as f:
            temp_path = f.name
            torch.save(self._to_dict(), temp_path)
        os.rename(temp_path, path)

    def load_ckpt(self, path: str, device: torch.device) -> None:
        state = torch.load(path, weights_only=True, map_location=device)
        self.global_step = state["global_step"]
        self.ddp.load_state_dict(state["ddp"])
        self.ema.load_state_dict(state["ema"])
        self.opt.load_state_dict(state["opt"])

    def save_latest_ckpt(self, dir: str) -> None:
        path = os.path.join(dir, f"step_{self.global_step}.ckpt")
        self.save_ckpt(path)

    def load_latest_ckpt(self, dir: str, device: torch.device) -> None:
        pattern = r"step_(\d+).ckpt"
        fnames = [
            f.name
            for f in os.scandir(dir)
            if f.is_file() and re.fullmatch(pattern, f.name)
        ]
        if len(fnames) == 0:
            print(f"No existing checkpoints found in {dir}, skipping load.")
            return None

        latest = max(fnames, key=lambda x: int(re.fullmatch(pattern, x).group(1)))  # type: ignore
        path = os.path.join(dir, latest)
        print(f"Found previous checkpoint, loading latest from {path}")
        self.load_ckpt(path, device)


def guided_prediction(model: DiTWrapper, x, t, y, e, c, cfg=1.0) -> torch.Tensor:
    if t.ndim == 0:  # the ODE solver gives scalar t
        t = t.expand(x.shape[0])

    if cfg == 0.0:  # unconditional
        return model(x=x, t=t, y=None, e=e, c=c)
    if cfg == 1.0:  # conditional
        return model(x=x, t=t, y=y, e=e, c=c)

    pred_cond = model(x=x, t=t, y=y, e=e, c=c)
    pred_null = model(x=x, t=t, y=None, e=e, c=c)
    return pred_null + cfg * (pred_cond - pred_null)


@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def generate(
    *,
    model: DiTWrapper,  # DiT, maps x,cond -> velocity
    x0: torch.Tensor | None = None,  # shape (B, C, H, W), from source distribution
    y1: torch.Tensor,  # shape (B,) - perturbation condition
    e1: torch.Tensor,  # shape (B,) - experiment condition
    c1: torch.Tensor,  # shape (B,) - cell type
) -> tuple[torch.Tensor, int]:
    """Generate a sample with the dopri5 probability flow ODE solver."""

    nfe = 0  # NB, if using guidance, the true nfe is this * 2

    def forward_fn(t, x):
        nonlocal nfe
        nfe += 1
        return guided_prediction(model, x=x, t=t, y=y1, e=e1, c=c1)

    traj = torchdiffeq.odeint(
        forward_fn,
        x0,
        torch.linspace(0, 1, 2, device=y1.device),
        method="dopri5",
        rtol=1e-5,
        atol=1e-5,
    )
    return traj[-1], nfe  # type: ignore


@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def compute_loss(
    model: DDP,  # DDP wrapper around DiT, maps t,x,y,e -> velocity
    x0: torch.Tensor,  # shape (B, C, H, W), sampled from source distribution
    x1: torch.Tensor,  # shape (B, C, H, W), sampled from target distribution
    y1: torch.Tensor,  # shape (B,) - [0,perturbations)
    e1: torch.Tensor,  # shape (B,) - experiment condition
    c1: torch.Tensor,  # shape (B,) - cell type
) -> torch.Tensor:
    t = torch.rand_like(y1, dtype=torch.float32)  # shape (B,) - [0,1)
    t_ = t.reshape(-1, 1, 1, 1).expand_as(x0)  # B -> B,C,H,W

    alpha_t = t_
    d_alpha_t = 1.0
    sigma_t = 1.0 - t_
    d_sigma_t = -1.0

    xt = alpha_t * x1 + sigma_t * x0
    ut = d_alpha_t * x1 + d_sigma_t * x0

    vt = model(x=xt, t=t, y=y1, e=e1, c=c1)
    return torch.nn.functional.mse_loss(vt, ut)


@torch.inference_mode()
def evaluate(
    *,
    state: TrainState,
    loader: torch.utils.data.DataLoader,
    D: Distributed,
    use_ema: bool,
) -> None:
    model: DiTWrapper = state.ema.module if use_ema else state.ddp.module  # type: ignore
    model.eval()

    running_loss = torch.tensor(0.0, device=D.device)
    num_samples = torch.tensor(0, device=D.device)

    for batch in tqdm(
        loader,
        desc="Evaluating",
        total=len(loader),
        unit="step",
        disable=D.rank != 0,  # only show from one process
    ):
        x1 = batch["pert"].to(D.device, non_blocking=True)
        y1 = batch["perturbation_id"].to(D.device, non_blocking=True)
        exp = batch["experiment_id"].to(D.device, non_blocking=True)
        c1 = batch["cell_type_id"].to(D.device, non_blocking=True)
        x0 = torch.randn_like(x1)

        loss = compute_loss(
            model=model,  # type: ignore
            x0=x0,
            x1=x1,
            y1=y1,
            e1=exp,
            c1=c1,
        )
        running_loss += loss * x1.shape[0]
        num_samples += x1.shape[0]

    running_loss = D.all_reduce(running_loss, op="sum")
    num_samples = D.all_reduce(num_samples, op="sum")

    val_loss = (running_loss / num_samples).item()
    prefix = "ema" if use_ema else "ddp"
    log(step=state.global_step, data={f"val/{prefix}_loss": val_loss})
    D.barrier()
    return


@rank_zero()
@torch.inference_mode()
def visualise(
    *,
    state: TrainState,
    loader: torch.utils.data.DataLoader,
    output_dir: str,
    D: Distributed,
    use_ema: bool,
    mode: Literal["generation", "counterfactual"],
):
    model: DiTWrapper = state.ema.module if use_ema else state.ddp.module  # type: ignore
    model.eval()

    batch = next(iter(loader))
    x1 = batch["pert"].to(D.device, non_blocking=True)
    y1 = batch["perturbation_id"].to(D.device, non_blocking=True)
    exp = batch["experiment_id"].to(D.device, non_blocking=True)
    c1 = batch["cell_type_id"].to(D.device, non_blocking=True)
    x0 = torch.randn_like(x1)

    preds, nfe = generate(model=model, x0=x0, y1=y1, e1=exp, c1=c1)
    preds = decode(preds)  # uint8 [B, 3, H, W]
    preds = preds.to(torch.float32) / 255  # save_image wants float32

    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    save_path = os.path.join(output_dir, f"samples/{state.global_step}.png")
    save_image(preds, save_path)
    prefix = "ema" if use_ema else "ddp"
    log(
        step=state.global_step,
        data={f"val/{prefix}_samples": wandb.Image(save_path), "val/nfe": nfe},
    )
    return


@torch.inference_mode()
def compute_metrics(
    *,
    state: TrainState,
    name: str,
    loader: torch.utils.data.DataLoader,
    D: Distributed,
    use_ema: bool,
    mode: Literal["generation", "counterfactual"],
) -> None:
    model: DiTWrapper = state.ema.module if use_ema else state.ddp.module  # type: ignore
    model.eval()

    detector = FeatureExtractor(D.device)
    inception_features_real = []
    inception_features_fake = []
    dino_features_real = []
    dino_features_fake = []

    mean_nfe = torch.tensor(0.0)

    N = len(loader.dataset)  # type: ignore
    log(step=state.global_step, msg=f"Generating {N} samples for {name}")

    disable_tqdm: bool = D.rank != 0
    for _, batch in enumerate(
        tqdm(
            loader,
            desc=f"Computing metrics for {name}",
            total=len(loader),
            unit="step",
            disable=disable_tqdm,  # only show from one process
        )
    ):
        x1 = batch["pert"].to(D.device, non_blocking=True)
        y1 = batch["perturbation_id"].to(D.device, non_blocking=True)
        exp = batch["experiment_id"].to(D.device, non_blocking=True)
        c1 = batch["cell_type_id"].to(D.device, non_blocking=True)
        x0 = torch.randn_like(x1)

        preds, nfe = generate(model=model, x0=x0, y1=y1, e1=exp, c1=c1)
        mean_nfe += nfe
        preds = decode(preds)  # uint8 [B, 3, H, W]
        x1 = decode(x1)

        inception_fake, dino_fake = detector(preds)
        inception_real, dino_real = detector(x1)

        inception_features_real.append(inception_real.cpu())
        inception_features_fake.append(inception_fake.cpu())
        dino_features_real.append(dino_real.cpu())
        dino_features_fake.append(dino_fake.cpu())

    D.barrier()

    # move to gpu, all_gather, move to cpu
    # we do it this way because all_gather is not supported on CPU
    def gather_cpu(x) -> np.ndarray:
        x = torch.cat(x, dim=0).to(D.device)
        x = D.gather_concat(x).cpu().numpy()
        return x[:N]  # truncate duplicates from last batch of distributed dataloader

    inception_features_real = gather_cpu(inception_features_real)
    inception_features_fake = gather_cpu(inception_features_fake)
    dino_features_real = gather_cpu(dino_features_real)
    dino_features_fake = gather_cpu(dino_features_fake)

    mean_nfe /= len(loader)
    mean_nfe = D.all_reduce(mean_nfe.to(D.device), op="avg").cpu().item()

    if D.rank == 0:
        prefix = "ema" if use_ema else "ddp"
        fd_dino = compute_fd(dino_features_real, dino_features_fake)
        fd_inception = compute_fd(inception_features_real, inception_features_fake)
        kd_dino = compute_kd(dino_features_real, dino_features_fake)
        kd_inception = compute_kd(inception_features_real, inception_features_fake)
        log(
            step=state.global_step,
            data={
                f"{name}_metrics/{prefix}_fd_dinov2": fd_dino,
                f"{name}_metrics/{prefix}_kd_dinov2": kd_dino,
                f"{name}_metrics/{prefix}_fid": fd_inception,
                f"{name}_metrics/{prefix}_kid": kd_inception,
                f"{name}_metrics/{prefix}_nfe_per_batch": mean_nfe,
            },
        )

    D.barrier()


def train(
    *,
    state: TrainState,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loaders: dict[str, torch.utils.data.DataLoader],
    output_dir: str,
    D: Distributed,
    num_steps: int = 100_000,
    log_every_n_steps: int = 250,
    eval_every_n_steps: int = 5000,
    ckpt_every_n_steps: int = 25_000,
    metrics_every_n_steps: int = 25_000,
) -> None:
    prof = get_profiler(return_dummy=D.rank != 0, save_dir=output_dir)
    start_step = state.global_step
    loader = infinite_dataloader(train_loader, start_step=start_step, fast_resume=True)

    @torch.compile(fullgraph=False)
    def step(step, opt, ema, ddp):  # compiling the optimiser and ema step is helpful
        opt.step()
        opt.zero_grad()
        ema.update(model=ddp.module, step=step)

    running_loss = torch.tensor(0.0, device=D.device)
    num_samples = torch.tensor(0, device=D.device)
    start_time = time.time()
    prof.start()
    for _, batch in tqdm(
        zip(range(start_step, num_steps), loader),
        desc="Training",
        initial=start_step,
        total=num_steps,
        unit="step",
        disable=D.rank != 0,  # only show from one process
    ):
        state.ddp.train()
        x1 = batch["pert"].to(D.device, non_blocking=True)
        y1 = batch["perturbation_id"].to(D.device, non_blocking=True)
        exp = batch["experiment_id"].to(D.device, non_blocking=True)
        c1 = batch["cell_type_id"].to(D.device, non_blocking=True)
        x0 = torch.randn_like(x1)

        loss = compute_loss(model=state.ddp, x0=x0, x1=x1, y1=y1, e1=exp, c1=c1)
        running_loss += loss.detach() * x0.shape[0]  # total batch loss
        num_samples += x0.shape[0]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(state.ddp.parameters(), max_norm=0.5)

        state.global_step += 1
        step(state.global_step, state.opt, state.ema, state.ddp)
        prof.step()

        if state.global_step % log_every_n_steps == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            throughput_per_gpu = num_samples / elapsed_time

            running_loss = D.all_reduce(running_loss, op="sum")
            num_samples = D.all_reduce(num_samples, op="sum")

            throughput_total = num_samples / elapsed_time
            loss = (running_loss / num_samples).item()
            running_loss = torch.tensor(0.0, device=D.device)
            num_samples = torch.tensor(0, device=D.device)

            log(
                step=state.global_step,
                data={
                    "train/loss": loss,
                    "train/throughput_total": throughput_total,
                    "train/throughput_per_gpu": throughput_per_gpu,
                    "train/grad_norm": grad_norm,
                },
            )

        if state.global_step % eval_every_n_steps == 0:
            visualise(
                state=state,
                loader=val_loader,
                output_dir=output_dir,
                D=D,
                use_ema=True,
                mode="generation",
            )
            evaluate(state=state, loader=val_loader, D=D, use_ema=False)
            evaluate(state=state, loader=val_loader, D=D, use_ema=True)

        if state.global_step % metrics_every_n_steps == 0:
            for name, loader in test_loaders.items():
                compute_metrics(
                    state=state,
                    name=name,
                    loader=loader,
                    D=D,
                    use_ema=True,
                    mode="generation",
                )

        if state.global_step % ckpt_every_n_steps == 0:
            ckpt_dir = os.path.join(output_dir, "checkpoints")
            log(step=state.global_step, msg=f"Saving checkpoint to {ckpt_dir}")
            state.save_latest_ckpt(dir=ckpt_dir)

    log(step=state.global_step, msg="Training complete")
    return
