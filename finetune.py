"""Finetune an AVC model on BBBC021 using Morgan fingerprint embeddings.
This involves training a small 1-token transformer (essentially a fancy MLP)
to map from Morgan fingerprints to the DiT's label embedding space.
This enables the model to generalise to unseen perturbations at inference time.
"""

import gc
import logging
import os
import pathlib

import numpy as np
import ornamentalist
import submitit
import torch
import torch.nn as nn
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

import bbbc021
import model
import trainer
from utils.distributed import Distributed
from utils.ema import KarrasEMA
from utils.name_run import generate_random_name

# You will need to fill in these constants appropriately
CKPT_PATH: str = ...  #  path to the pretrained AVC checkpoint from main.py
EMBS_ARRAY: np.ndarray = ...  # shape (num_perts, 2048) - maps pert to fingerprint
DOSE_ARRAY: np.ndarray = ...  #  shape (num_perts,) - maps pert to dose

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class MorganFingerprintEmbedder(nn.Module):
    def __init__(self, hidden_size=1152, depth=6, num_heads=6, mlp_ratio=1.0):
        super().__init__()
        self.in_proj = nn.Linear(2048, hidden_size)  # Morgan dim is 2048
        self.blocks = nn.ModuleList(
            [
                model.DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        self.dose_embedder = model.ScalarEmbedder(
            hidden_size, frequency_embedding_size=256
        )

        embs_array = torch.from_numpy(EMBS_ARRAY).to(torch.float32)
        dose_array = torch.from_numpy(DOSE_ARRAY).to(torch.float32)

        # standardise dose array to 0,1
        dose_array = (dose_array - dose_array.min()) / (
            dose_array.max() - dose_array.min()
        )

        self.register_buffer("embs_array", embs_array)
        self.register_buffer("dose_array", dose_array)

    def forward(self, y, train):
        del train  # train is just to make the signature match LabelEmbedder

        emb = self.embs_array[y]  # type: ignore - (B, 2048)
        dose = self.dose_array[y]  # type: ignore - (B,)

        dose = self.dose_embedder(dose)  # B, hidden_size
        emb = self.in_proj(emb).unsqueeze(1)  # B, 1, hidden_size
        for block in self.blocks:
            emb = block(emb, dose)
        return emb.squeeze(1)  # B, hidden_size


def load_ckpt(state: trainer.TrainState, D: Distributed):
    log.info(f"Loading checkpoint from {CKPT_PATH}")
    state.load_ckpt(path=CKPT_PATH, device=D.device)


def prng(rank: int, seed: int = 42):
    local_seed = seed + rank
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed(local_seed)
    log.info(f"Rank {rank} setting seed to {local_seed}")


def main(config: ornamentalist.ConfigDict):
    ornamentalist.setup(config, force=True)
    with Distributed() as D:
        job_env = submitit.JobEnvironment()

        cwd = pathlib.Path.cwd()
        output_dir = str(cwd.parent / job_env.job_id)
        name = generate_random_name(output_dir)

        if D.rank == 0:
            wandb.init(
                name=name,
                id=name,
                dir=output_dir,
                notes=f"Outputs saved to: {output_dir}",
                project="sota-hunting",
                entity="valencelabs",
                resume="allow",
                save_code=False,
                config=config,
            )

        log.info(f"Running job ID {job_env.job_id}")
        log.info(f"{output_dir=}")
        log.info(f"{name=}")
        log.info(f"{config=}")

        prng(D.rank)
        torch.set_float32_matmul_precision("medium")

        # --- first create a dummy model and TrainState ---
        model_cls = model.get_model_cls(name="DiT-XL/2")
        net = model_cls(
            in_channels=bbbc021.IMG_CHANNELS,
            input_size=bbbc021.IMG_SIZE,
            y_dim=bbbc021.NUM_PERTURBATIONS,
            e_dim=bbbc021.NUM_EXPERIMENTS,
            c_dim=bbbc021.NUM_CELL_TYPES,
        )
        net.to(D.device)
        ddp = DDP(net)
        ema = KarrasEMA(net)
        opt = torch.optim.Adam(
            net.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        state = trainer.TrainState(ddp=ddp, ema=ema, opt=opt, global_step=0)

        # --- load pretrained weights into the dummy model ---
        load_ckpt(state=state, D=D)

        # --- now do model surgery to replace the label embedder with the Morgan fingerprint embedder ---
        net = state.ema.module
        assert isinstance(net, model.DiTWrapper)
        del state
        del opt
        del ema
        del ddp
        del net.adaptor.y_embedder
        gc.collect()
        torch.cuda.empty_cache()

        embedder = MorganFingerprintEmbedder()
        net.adaptor.y_embedder = embedder  # type: ignore
        net.to(D.device)

        # --- only train the embedder and adaLN modulation layers ---

        # freeze net
        for param in net.parameters():
            param.requires_grad = False

        # unfreeze embedder
        for param in net.adaptor.y_embedder.parameters():
            param.requires_grad = True

        # unfreeze adaLN modulation in DiT blocks
        for block in net.dit.blocks:
            for param in block.adaLN_modulation.parameters():  # type: ignore
                param.requires_grad = True

        log.info(
            f"Trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6:.2f}M"
        )

        # --- re-make TrainState and kick off training ---

        net.compile()
        ddp = DDP(net)
        ema = KarrasEMA(net)
        opt = torch.optim.Adam(
            net.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        state = trainer.TrainState(ddp=ddp, ema=ema, opt=opt, global_step=0)

        gc.collect()
        torch.cuda.empty_cache()

        ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        log.info(f"Using checkpoint directory: {ckpt_dir}")

        train_loader, iid_loader, ood_loader = bbbc021.get_dataloaders()
        trainer.train(
            state=state,
            train_loader=train_loader,
            val_loader=iid_loader,
            test_loaders={"iid": iid_loader, "ood": ood_loader},
            output_dir=output_dir,
            D=D,
        )

        if D.rank == 0:
            wandb.finish(quiet=True)
