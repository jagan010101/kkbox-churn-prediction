"""Training loop (Section 6.1, used by Exp-1..5) and the two learned/gradient-surgery
multi-task variants (Exp-6 uncertainty weighting, Exp-7 PCGrad).

Extracted from 04_Training_Baselines.ipynb (run_epoch/train_model) and
05_MultiTask_Ablation.ipynb (train_uncertainty_weighted, PCGrad, train_pcgrad).
Numerics are unchanged from those notebooks; only module-level globals
(CAT_COLS, train_loader, EPOCHS, ...) became explicit parameters.

Exp-3/4/5 (05's fixed loss-weight sweep) previously had their own
train_fixed_weight with a smaller history schema than 04's train_model, but
the two compute mathematically identical val_loss (linearity of the
lambda-weighted average lets you swap the order of weighting and averaging)
- so here they share train_model, gaining a few extra history columns
(train_loss, lr) rather than carrying a second near-duplicate function.
"""

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def gather_predictions(model, loader):
    model.eval()
    all_logits, all_churn, all_ltv_pred, all_ltv_true = [], [], [], []
    with torch.no_grad():
        for x_num, x_cat, y_churn, y_ltv in loader:
            churn_logit, ltv_pred = model(x_num, x_cat)
            all_logits.append(churn_logit)
            all_churn.append(y_churn)
            all_ltv_pred.append(ltv_pred)
            all_ltv_true.append(y_ltv)
    return torch.cat(all_logits), torch.cat(all_churn), torch.cat(all_ltv_pred), torch.cat(all_ltv_true)


def eval_metrics(model, loader):
    """AUC-ROC (churn) and RMSE (log1p LTV) on a loader."""
    logits, churn_true, ltv_pred, ltv_true = gather_predictions(model, loader)
    auc = roc_auc_score(churn_true.numpy(), torch.sigmoid(logits).numpy())
    rmse = torch.sqrt(torch.mean((ltv_pred - ltv_true) ** 2)).item()
    return auc, rmse


def run_epoch(model, loader, bce_fn, mse_fn, lambda_churn, lambda_ltv, optimizer=None, grad_clip_norm=1.0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, n = 0.0, 0
    all_logits, all_churn, all_ltv_pred, all_ltv_true = [], [], [], []
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x_num, x_cat, y_churn, y_ltv in loader:
            if is_train:
                optimizer.zero_grad()
            churn_logit, ltv_pred = model(x_num, x_cat)
            loss = lambda_churn * bce_fn(churn_logit, y_churn) + lambda_ltv * mse_fn(ltv_pred, y_ltv)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            total_loss += loss.item() * len(y_churn)
            n += len(y_churn)
            all_logits.append(churn_logit.detach())
            all_churn.append(y_churn)
            all_ltv_pred.append(ltv_pred.detach())
            all_ltv_true.append(y_ltv)
    logits, churn_true = torch.cat(all_logits), torch.cat(all_churn)
    ltv_pred_all, ltv_true_all = torch.cat(all_ltv_pred), torch.cat(all_ltv_true)
    auc = roc_auc_score(churn_true.numpy(), torch.sigmoid(logits).numpy())
    rmse_log = torch.sqrt(torch.mean((ltv_pred_all - ltv_true_all) ** 2)).item()
    return total_loss / n, auc, rmse_log


def train_model(model, train_loader, val_loader, lambda_churn, lambda_ltv, pos_weight, train_cfg,
                 checkpoint_path=None, verbose=True):
    """Single fixed loss-weight combo: Exp-1 (1,0), Exp-2 (0,1), Exp-3..5 (mixed)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=train_cfg["lr_scheduler_patience"], factor=train_cfg["lr_scheduler_factor"]
    )
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse_fn = nn.MSELoss()

    best_val_loss, epochs_no_improve, history = float("inf"), 0, []
    for epoch in range(train_cfg["epochs"]):
        tr_loss, tr_auc, tr_rmse = run_epoch(
            model, train_loader, bce_fn, mse_fn, lambda_churn, lambda_ltv, optimizer, train_cfg["grad_clip_norm"]
        )
        val_loss, val_auc, val_rmse = run_epoch(model, val_loader, bce_fn, mse_fn, lambda_churn, lambda_ltv)
        scheduler.step(val_loss)
        cur_lr = optimizer.param_groups[0]["lr"]
        history.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=val_loss, val_auc=val_auc,
                             val_rmse_log=val_rmse, lr=cur_lr))
        if verbose:
            print(f"epoch {epoch:2d} train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_auc={val_auc:.4f} val_rmse_log={val_rmse:.4f} lr={cur_lr:.2e}")
        if val_loss < best_val_loss - 1e-5:
            best_val_loss, epochs_no_improve = val_loss, 0
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= train_cfg["patience"]:
                if verbose:
                    print(f"early stopping at epoch {epoch}")
                break
    return pd.DataFrame(history)


def train_uncertainty_weighted(model, train_loader, val_loader, train_cfg, checkpoint_path=None, verbose=True):
    """Exp-6: Kendall, Gal & Cipolla 2018. Learns log-variance per task;
    L = 0.5*exp(-s)*L_task + 0.5*s for each task, summed.
    """
    log_var_churn = nn.Parameter(torch.zeros(1))
    log_var_ltv = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [log_var_churn, log_var_ltv],
        lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=train_cfg["lr_scheduler_patience"], factor=train_cfg["lr_scheduler_factor"]
    )
    bce_fn = nn.BCEWithLogitsLoss()
    mse_fn = nn.MSELoss()

    best_val_loss, epochs_no_improve, history = float("inf"), 0, []
    for epoch in range(train_cfg["epochs"]):
        model.train()
        for x_num, x_cat, y_churn, y_ltv in train_loader:
            optimizer.zero_grad()
            churn_logit, ltv_pred = model(x_num, x_cat)
            loss = (
                0.5 * torch.exp(-log_var_churn) * bce_fn(churn_logit, y_churn) + 0.5 * log_var_churn
                + 0.5 * torch.exp(-log_var_ltv) * mse_fn(ltv_pred, y_ltv) + 0.5 * log_var_ltv
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip_norm"])
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_bce, val_mse, n = 0.0, 0.0, 0
            for x_num, x_cat, y_churn, y_ltv in val_loader:
                churn_logit, ltv_pred = model(x_num, x_cat)
                val_bce += bce_fn(churn_logit, y_churn).item() * len(y_churn)
                val_mse += mse_fn(ltv_pred, y_ltv).item() * len(y_churn)
                n += len(y_churn)
        val_bce, val_mse = val_bce / n, val_mse / n
        val_loss = (
            0.5 * torch.exp(-log_var_churn).item() * val_bce + 0.5 * log_var_churn.item()
            + 0.5 * torch.exp(-log_var_ltv).item() * val_mse + 0.5 * log_var_ltv.item()
        )
        val_auc, val_rmse = eval_metrics(model, val_loader)
        scheduler.step(val_loss)
        history.append(dict(epoch=epoch, val_loss=val_loss, val_auc=val_auc, val_rmse_log=val_rmse,
                             log_var_churn=log_var_churn.item(), log_var_ltv=log_var_ltv.item()))
        if verbose:
            print(f"epoch {epoch:2d} val_loss={val_loss:.4f} val_auc={val_auc:.4f} val_rmse_log={val_rmse:.4f} "
                  f"log_var_churn={log_var_churn.item():.3f} log_var_ltv={log_var_ltv.item():.3f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss, epochs_no_improve = val_loss, 0
            if checkpoint_path:
                torch.save({"model": model.state_dict(), "log_var_churn": log_var_churn.detach(),
                            "log_var_ltv": log_var_ltv.detach()}, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= train_cfg["patience"]:
                if verbose:
                    print(f"early stopping at epoch {epoch}")
                break
    return model, pd.DataFrame(history)


class PCGrad:
    """Yu et al., NeurIPS 2020. Projects away each task-pair's conflicting
    gradient component (negative cosine similarity) before summing.
    """

    def __init__(self, optimizer, params):
        self.optimizer = optimizer
        self.params = list(params)

    def pc_backward(self, losses):
        grads_per_task = []
        for loss in losses:
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            grads_per_task.append([p.grad.clone() if p.grad is not None else None for p in self.params])

        num_tasks = len(losses)
        projected = [list(g) for g in grads_per_task]
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i == j:
                    continue
                for k in range(len(self.params)):
                    g_i = projected[i][k]
                    g_j = grads_per_task[j][k]  # project against the ORIGINAL other-task grad
                    if g_i is None or g_j is None:
                        continue
                    g_i_flat, g_j_flat = g_i.flatten(), g_j.flatten()
                    dot = torch.dot(g_i_flat, g_j_flat)
                    if dot < 0:
                        g_i_flat = g_i_flat - (dot / (g_j_flat.dot(g_j_flat) + 1e-12)) * g_j_flat
                        projected[i][k] = g_i_flat.view_as(g_i)

        final_grads = []
        for k in range(len(self.params)):
            total = None
            for i in range(num_tasks):
                g = projected[i][k]
                if g is None:
                    continue
                total = g if total is None else total + g
            final_grads.append(total)

        self.optimizer.zero_grad()
        for p, g in zip(self.params, final_grads):
            p.grad = g

    def step(self):
        self.optimizer.step()


def train_pcgrad(model, train_loader, val_loader, train_cfg, checkpoint_path=None, verbose=True):
    """Exp-7: PCGrad. No scalar lambda - val_loss reported as the unweighted BCE+MSE sum."""
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=train_cfg["lr_scheduler_patience"], factor=train_cfg["lr_scheduler_factor"]
    )
    bce_fn = nn.BCEWithLogitsLoss()
    mse_fn = nn.MSELoss()
    pcgrad = PCGrad(optimizer, model.parameters())

    best_val_loss, epochs_no_improve, history = float("inf"), 0, []
    for epoch in range(train_cfg["epochs"]):
        model.train()
        for x_num, x_cat, y_churn, y_ltv in train_loader:
            churn_logit, ltv_pred = model(x_num, x_cat)
            loss_churn = bce_fn(churn_logit, y_churn)
            loss_ltv = mse_fn(ltv_pred, y_ltv)
            pcgrad.pc_backward([loss_churn, loss_ltv])
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip_norm"])
            pcgrad.step()

        model.eval()
        with torch.no_grad():
            val_bce, val_mse, n = 0.0, 0.0, 0
            for x_num, x_cat, y_churn, y_ltv in val_loader:
                churn_logit, ltv_pred = model(x_num, x_cat)
                val_bce += bce_fn(churn_logit, y_churn).item() * len(y_churn)
                val_mse += mse_fn(ltv_pred, y_ltv).item() * len(y_churn)
                n += len(y_churn)
        val_bce, val_mse = val_bce / n, val_mse / n
        val_loss = val_bce + val_mse
        val_auc, val_rmse = eval_metrics(model, val_loader)
        scheduler.step(val_loss)
        history.append(dict(epoch=epoch, val_loss=val_loss, val_auc=val_auc, val_rmse_log=val_rmse))
        if verbose:
            print(f"epoch {epoch:2d} val_loss={val_loss:.4f} val_auc={val_auc:.4f} val_rmse_log={val_rmse:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss, epochs_no_improve = val_loss, 0
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= train_cfg["patience"]:
                if verbose:
                    print(f"early stopping at epoch {epoch}")
                break
    return model, pd.DataFrame(history)
