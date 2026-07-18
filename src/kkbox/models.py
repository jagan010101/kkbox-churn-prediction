"""Factorization Machine interaction layer + the multi-task FM-MLP.

Extracted verbatim from 03_Model_Architecture.ipynb (and re-defined
identically in 04-07 prior to this refactor).
"""

import torch
import torch.nn as nn


class FMInteractionLayer(nn.Module):
    """Rendle 2010: all pairwise feature interactions in O(k*d) via the
    sum-of-squares-minus-square-of-sums identity.
    """

    def __init__(self, input_dim, k=8):
        super().__init__()
        self.V = nn.Parameter(torch.randn(input_dim, k) * 0.01)

    def forward(self, x):
        xV = x.unsqueeze(2) * self.V.unsqueeze(0)
        sum_then_sq = xV.sum(dim=1).pow(2)
        sq_then_sum = xV.pow(2).sum(dim=1)
        return 0.5 * (sum_then_sq - sq_then_sum)


class MultiTaskFMNet(nn.Module):
    """Embeddings + numerical features -> FM interaction layer -> shared MLP
    backbone -> two independent heads (churn logit, log1p_ltv regression).
    """

    def __init__(self, cat_cols, cardinalities, embed_dims, num_numerical,
                 fm_k=8, backbone_dims=(256, 128, 64), dropouts=(0.3, 0.3, 0.2), head_hidden_dim=32):
        super().__init__()
        self.cat_cols = cat_cols
        self.embeddings = nn.ModuleDict(
            {col: nn.Embedding(cardinalities[col], embed_dims[col]) for col in cat_cols}
        )
        combined_dim = sum(embed_dims[c] for c in cat_cols) + num_numerical
        self.fm = FMInteractionLayer(combined_dim, k=fm_k)

        backbone_input = combined_dim + fm_k
        layers, prev = [], backbone_input
        for dim, p in zip(backbone_dims, dropouts):
            layers += [nn.Linear(prev, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(p)]
            prev = dim
        self.backbone = nn.Sequential(*layers)

        self.churn_head = nn.Sequential(nn.Linear(prev, head_hidden_dim), nn.ReLU(), nn.Linear(head_hidden_dim, 1))
        self.ltv_head = nn.Sequential(nn.Linear(prev, head_hidden_dim), nn.ReLU(), nn.Linear(head_hidden_dim, 1))

    def forward(self, x_num, x_cat):
        embeds = [self.embeddings[col](x_cat[:, i]) for i, col in enumerate(self.cat_cols)]
        x = torch.cat(embeds + [x_num], dim=1)
        fm_out = self.fm(x)
        h = torch.cat([x, fm_out], dim=1)
        shared = self.backbone(h)
        return self.churn_head(shared).squeeze(-1), self.ltv_head(shared).squeeze(-1)


def build_model(cardinalities, embed_dims, cat_cols, num_numerical, model_cfg):
    """Builds a MultiTaskFMNet from the 'model' section of config.yaml."""
    return MultiTaskFMNet(
        cat_cols=cat_cols,
        cardinalities=cardinalities,
        embed_dims=embed_dims,
        num_numerical=num_numerical,
        fm_k=model_cfg["fm_k"],
        backbone_dims=tuple(model_cfg["backbone_dims"]),
        dropouts=tuple(model_cfg["dropouts"]),
        head_hidden_dim=model_cfg["head_hidden_dim"],
    )
