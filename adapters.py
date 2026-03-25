import torch
import torch.nn as nn
import math

class ScalarEmbedder(nn.Module):
    """Embeds scalar conditions into vector representations. Useful for timesteps, zoom levels, etc."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations. Also handles dropout for cfg."""

    def __init__(self, num_classes, hidden_size, dropout_prob=0.15):
        super().__init__()
        # +1 for null token
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """Drops labels to enable classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class BasicAdaptor(nn.Module):
    """Adaptor module for conditioning layers in DiT generator.
    Uses OH embeddings (useful for pretraining but cannot generalise to new conditions).
    Assumes the following conditioning:
    - t: a scalar time condition, float range [0,1]
    - y: a class label, int range [0, y_dim-1]
    - e: an experiment label, int range [0, e_dim-1]
    - c: a cell type label, int range [0, c_dim-1]
    """

    def __init__(self, hidden_size, y_dim, e_dim, c_dim, frequency_embedding_size=256):
        super().__init__()
        self.t_embedder = ScalarEmbedder(hidden_size, frequency_embedding_size)
        self.y_embedder = LabelEmbedder(num_classes=y_dim, hidden_size=hidden_size)
        self.e_embedder = LabelEmbedder(num_classes=e_dim, hidden_size=hidden_size)
        self.c_embedder = LabelEmbedder(num_classes=c_dim, hidden_size=hidden_size)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)  # type: ignore
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)  # type: ignore
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.e_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.c_embedder.embedding_table.weight, std=0.02)

    def forward(self, t, y, e, c):
        t_emb = self.t_embedder(t)

        # allow passing None to use unconditional embedding for cfg
        if y is None:
            y = torch.full((t.shape[0],), self.y_embedder.num_classes, device=t.device)
        if e is None:
            e = torch.full((t.shape[0],), self.e_embedder.num_classes, device=t.device)
        if c is None:
            c = torch.full((t.shape[0],), self.c_embedder.num_classes, device=t.device)

        y_emb = self.y_embedder(y, train=self.training)
        e_emb = self.e_embedder(e, train=self.training)
        c_emb = self.c_embedder(c, train=self.training)
        return t_emb + y_emb + e_emb + c_emb
