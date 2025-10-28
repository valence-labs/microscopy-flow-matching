import numpy as np
import torch
from scipy import linalg
from torch_fidelity import FeatureExtractorInceptionV3
from torch_fidelity.interpolate_compat_tensorflow import (
    interpolate_bilinear_2d_like_tensorflow1x,
)
from torch_fidelity.metric_kid import kid_features_to_metric
from torchvision.transforms.functional import normalize


class FeatureExtractor:
    """Jointly extracts DinoV2 and Inception features for generative model evaluation."""

    def __init__(self, device: torch.device):
        self.dino = torch.hub.load(
            "facebookresearch/dinov2:main",
            "dinov2_vitl14",
            trust_repo=True,
            verbose=False,
            skip_validation=True,
        )
        assert isinstance(self.dino, torch.nn.Module)
        self.dino.eval().requires_grad_(False).to(device)

        self.inception = FeatureExtractorInceptionV3(
            name="InceptionNet", features_list=["2048"]
        )
        self.inception.eval().requires_grad_(False).to(device)

    def __call__(self, x):
        """Extract features using DinoV2 and inception.
        Input: torch.uint8 tensor of shape (B,3,H,W), range [0,255].
        Output: tuple[inception_features, dino_features]
        inception features are torch.float32, shape (B,2048).
        dino features are torch.float32, shape (B,1024).
        """

        inception_features = self.inception(x)[0]

        x = x.to(torch.float32)
        x = interpolate_bilinear_2d_like_tensorflow1x(
            x, size=(224, 224), align_corners=False
        )  # B x 3 x 224 x 224
        x = normalize(
            x,
            [255 * 0.485, 255 * 0.456, 255 * 0.406],
            [255 * 0.229, 255 * 0.224, 255 * 0.225],
            inplace=False,
        )
        dino_features = self.dino(x)  # type: ignore
        return inception_features, dino_features


def compute_kd(reps1, reps2):
    kid = kid_features_to_metric(
        torch.from_numpy(reps1),
        torch.from_numpy(reps2),
        kid_subset_size=100,
        verbose=False,
    )
    return kid["kernel_inception_distance_mean"]


def compute_statistics(reps):
    mu = np.mean(reps, axis=0)
    sigma = np.cov(reps, rowvar=False)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_2d(sigma)
    return mu, sigma


def compute_fd(reps1, reps2):
    mu1, sigma1 = compute_statistics(reps1)
    mu2, sigma2 = compute_statistics(reps2)
    sqrt_trace = np.real(linalg.eigvals(sigma1 @ sigma2) ** 0.5).sum()  # type: ignore
    return ((mu1 - mu2) ** 2).sum() + sigma1.trace() + sigma2.trace() - 2 * sqrt_trace
