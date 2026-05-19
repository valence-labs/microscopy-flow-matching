# Elucidating the Design Space of Flow Matching for Cellular Microscopy

This repo provides a recipe for training large flow-matching generative transformers on microscopy data. As of early 2026, it is the largest generative phenomics model to date, and attains SOTA performance for generating (both seen and unseen) perturbations on the public BBBC021 and RxRx1 benchmarks.

If you find this repo helpful, or would like to build on our methods, please cite our [preprint](https://arxiv.org/abs/2603.26790) :)

This repo includes code to train our MiT (a DiT with modifications to improve stability on microscopy data) on BBBC021 (including finetuning with Morgan fingerprints for generating unseen compound perturbations). The repo is intended to be minimal so you can copy-paste relevant parts into your own projects.

## Running

If you would like to directly run this code, please:

1. Clone this repo to an appropriate location.
1. Install the dependencies in a virtual environment (we recommend using `uv`).
1. Set up the BBBC021 dataset and set the metadata path in `bbbc021.py`.
1. Run `python main.py --help` to see view the command line options:

```
usage: main.py [-h] [--launcher.nodes  ...] [--launcher.gpus  ...]
               [--launcher.cpus  ...] [--launcher.ram  ...] [--launcher.timeout  ...] 
               [--launcher.output_dir  ...] [--launcher.cluster  ...] [--launcher.desc  ...]

options:
  -h, --help            show this help message and exit

launcher:
  Hyperparameters for utils.launcher.launch

  --launcher.nodes  ... Type: int (optional), default=1
  --launcher.gpus  ... Type: int (optional), default=1
  --launcher.cpus  ... Type: int (optional), default=12
  --launcher.ram  ... Type: int (optional), default=64
  --launcher.timeout  ... Type: int (optional), default=1440
  --launcher.output_dir  ... Type: str (optional), default=./outputs/
  --launcher.cluster  ... Type: str, choices: ('debug', 'local', 'slurm') (optional), default=debug
  --launcher.desc  ... Type: str (optional), default=""
```
To debug on your local machine, you can run:
```bash
python main.py --launcher.cluster debug
```

If you are on a SLURM cluster, run (from either a login node or an interactive job):
```bash
python main.py --launcher.cluster slurm --launcher.gpus 8 --launcher.nodes 2
```

This will queue a 16-GPU job, which should reproduce the MiT+One-Hot BBBC021 results for seen compunds.

To finetune with Morgan fingerprint embeddings for unseen compounds, first set the relevant global variables at the top of `finetune.py`, then run:
```bash
python finetune.py --launcher.cluster slurm --launcher.gpus 8
```
