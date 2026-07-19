"""Zero-Inflated LogNormal (ZILN) model for forward-revenue prediction.

A single network jointly predicts (p_logit, mu, log_sigma) for the
zero-inflated, right-skewed fwd_rev_59d target, trained with a combined
BCE + LogNormal-NLL loss (Wang et al., "A Deep Probabilistic Model for
Customer Lifetime Value Prediction", Google, 2019). This is the neural-net
challenger to the CatBoost Tweedie regressor in 03a_CatBoost_and_Cox_Models.ipynb
- see 03c_ZILN_ForwardRevenue.ipynb for the final comparison.

mu/log_sigma clamp ranges are fixed, not tuned: they're derived from the
empirical distribution of log(fwd_rev_59d) among payers (mean=5.180,
std=0.546, 1%=4.595, 99%=6.796, max=8.002), not from architecture search.
An earlier untuned clamp (log_sigma max=3.0, permitting sigma up to ~20)
produced technically-finite but astronomically large point estimates for a
handful of outlier examples that dominated RMSE (R^2=-42682) - the fix was
tightening the clamp to match the real data, not changing the formula.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from kkbox.determinism import seed_everything

MU_CLAMP = (-1.0, 9.0)
LOG_SIGMA_CLAMP = (-3.0, 0.7)  # sigma in [~0.05, ~2.0]


class ZILNNet(nn.Module):
    def __init__(self, cat_cols, cardinalities, embed_dims, num_numerical,
                 hidden_dim1=128, hidden_dim2=64, dropout1=0.2, dropout2=0.1):
        super().__init__()
        self.cat_cols = cat_cols
        self.embeddings = nn.ModuleDict(
            {col: nn.Embedding(cardinalities[col], embed_dims[col]) for col in cat_cols}
        )
        combined_dim = sum(embed_dims[c] for c in cat_cols) + num_numerical
        self.backbone = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim1), nn.BatchNorm1d(hidden_dim1), nn.ReLU(), nn.Dropout(dropout1),
            nn.Linear(hidden_dim1, hidden_dim2), nn.BatchNorm1d(hidden_dim2), nn.ReLU(), nn.Dropout(dropout2),
        )
        self.head = nn.Linear(hidden_dim2, 3)  # p_logit, mu, log_sigma

    def forward(self, x_num, x_cat):
        embeds = [self.embeddings[col](x_cat[:, i]) for i, col in enumerate(self.cat_cols)]
        x = torch.cat(embeds + [x_num], dim=1)
        h = self.backbone(x)
        out = self.head(h)
        p_logit, mu, log_sigma = out[:, 0], out[:, 1], out[:, 2]
        mu = torch.clamp(mu, min=MU_CLAMP[0], max=MU_CLAMP[1])
        log_sigma = torch.clamp(log_sigma, min=LOG_SIGMA_CLAMP[0], max=LOG_SIGMA_CLAMP[1])
        return p_logit, mu, log_sigma


def ziln_loss(y, p_logit, mu, log_sigma):
    is_pos = (y > 0).float()
    clf_loss = F.binary_cross_entropy_with_logits(p_logit, is_pos, reduction="none")
    sigma = torch.exp(log_sigma)
    safe_y = torch.where(y > 0, y, torch.ones_like(y))
    log_y = torch.log(safe_y)
    lognormal_nll = log_y + log_sigma + 0.5 * np.log(2 * np.pi) + (log_y - mu) ** 2 / (2 * sigma ** 2)
    return (clf_loss + is_pos * lognormal_nll).mean()


def ziln_predict(p_logit, mu, log_sigma):
    sigma = torch.exp(log_sigma)
    p = torch.sigmoid(p_logit)
    return (p * torch.exp(mu + sigma ** 2 / 2)).clamp(max=1e6)


def build_loaders(df_train, df_val, cat_cols, num_cols, batch_size):
    def to_tensors(df):
        x_cat = torch.tensor(df[cat_cols].values, dtype=torch.long)
        x_num = torch.tensor(df[num_cols].values, dtype=torch.float32)
        y = torch.tensor(df["fwd_rev_59d"].values, dtype=torch.float32)
        return TensorDataset(x_cat, x_num, y)

    train_loader = DataLoader(to_tensors(df_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(to_tensors(df_val), batch_size=4096, shuffle=False)
    return train_loader, val_loader


def train_one_model(df_train, df_val, cat_cols, num_cols, cardinalities, embed_dims,
                     hidden_dim1=128, hidden_dim2=64, dropout1=0.2, dropout2=0.1,
                     learning_rate=1e-3, weight_decay=1e-4, batch_size=2048,
                     max_epochs=40, patience=6, seed=42, verbose=True):
    """Trains a single ZILN model to convergence (early stopping on val ZILN loss).
    Returns (model, best_val_loss, best_val_rmse) with the best-epoch weights loaded.
    """
    seed_everything(seed)
    train_loader, val_loader = build_loaders(df_train, df_val, cat_cols, num_cols, batch_size)

    model = ZILNNet(cat_cols, cardinalities, embed_dims, len(num_cols),
                     hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, dropout1=dropout1, dropout2=dropout2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_val_loss, epochs_no_improve, best_state = float("inf"), 0, None
    for epoch in range(max_epochs):
        model.train()
        for x_cat, x_num, y in train_loader:
            optimizer.zero_grad()
            p_logit, mu, log_sigma = model(x_num, x_cat)
            loss = ziln_loss(y, p_logit, mu, log_sigma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_cat, x_num, y in val_loader:
                p_logit, mu, log_sigma = model(x_num, x_cat)
                val_losses.append(ziln_loss(y, p_logit, mu, log_sigma).item())
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        improved = val_loss < best_val_loss - 1e-4
        if improved:
            best_val_loss, epochs_no_improve = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
        if verbose and (epoch % 5 == 0 or improved):
            print(f"  [seed {seed}] epoch {epoch:3d}  val_ziln_loss={val_loss:.4f}{' *' if improved else ''}")
        if epochs_no_improve >= patience:
            if verbose:
                print(f"  [seed {seed}] early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()

    rmse = float(np.sqrt(np.mean((predict_df(model, df_val, cat_cols, num_cols) - df_val["fwd_rev_59d"].values) ** 2)))
    return model, best_val_loss, rmse


@torch.no_grad()
def predict_df(model, df, cat_cols, num_cols, batch_size=8192):
    """Runs ziln_predict over a dataframe in batches, returns a numpy array of raw-TWD predictions."""
    model.eval()
    x_cat = torch.tensor(df[cat_cols].values, dtype=torch.long)
    x_num = torch.tensor(df[num_cols].values, dtype=torch.float32)
    preds = []
    for i in range(0, len(df), batch_size):
        p_logit, mu, log_sigma = model(x_num[i:i + batch_size], x_cat[i:i + batch_size])
        preds.append(ziln_predict(p_logit, mu, log_sigma))
    return torch.cat(preds).numpy()


def predict_ensemble(models, df, cat_cols, num_cols, batch_size=8192):
    """Averages predict_df() across a list of independently-seeded models -
    a standard ZILN robustness technique, since a single model's sigma estimate
    (and therefore its exp(mu + sigma^2/2) point estimate) can be noisy per-seed.
    """
    preds = np.stack([predict_df(m, df, cat_cols, num_cols, batch_size) for m in models])
    return preds.mean(axis=0)
