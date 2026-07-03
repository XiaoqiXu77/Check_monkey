#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RNN gauge predictor (incremental belief) + analyses:
  - Sanity check: true vs baseline vs rate-lesion predictions (MAP, discrete)
  - Progress-rate decoding vs check number (k=0 baseline = pre-first-check hidden)
  - PCA of hidden states colored by progress rate (refit per condition)
  - Cross-rate feedback decoding (4x4 accuracy matrix, 50-100% colorbar)
  - Cosine similarity of feedback axes: dot clouds for all cross-rate block pairs + mean off-diagonal
  - Lesion MSE vs #lesioned units:
      baseline vs top-k rate lesion vs random (median + 5–95%)
      IMPORTANT: ONLY here, gs observation is masked after 2 checks (max_obs_checks=2)

Key modeling constraint:
  - Predict gauge belief BEFORE observing gs on check trials (no leakage).
  - gs input channel exists always, but carries info only on check trials (and only first 2 checks for lesion MSE test).
  - No hard-coded rule to ignore incorrect fb; instead we add a penalty discouraging increments on incorrect trials.
  - Rate labels are 1..4 (corresponding to raw rates 2..5).
"""

import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


# -----------------------------
# Style / Reproducibility
# -----------------------------
def set_pub_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2.4,
        "axes.linewidth": 1.2,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.8,
    })


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Data: blocks
# -----------------------------
@dataclass
class Block:
    session_id: int
    fb: np.ndarray       # 0 check, 1 correct work, 2 incorrect work
    gs: np.ndarray       # gauge size (1..7), start-of-trial
    choice: np.ndarray   # 0 check, 1 work
    rate_raw: int        # 2..5
    rate_label: int      # 1..4 (rate_raw - 1)
    n_checks: int


def _to_int_gs(gs: np.ndarray) -> np.ndarray:
    g = np.rint(gs).astype(int)
    return np.clip(g, 1, 7)


def compute_progress_rate_label(block_fb: np.ndarray, block_gs: np.ndarray) -> int:
    """
    Count correct work trials until the first trial where gs reaches 7 (inclusive),
    divide by 7, round -> rate_raw in {2,3,4,5}.
    """
    idx7 = np.where(block_gs >= 7)[0]
    if len(idx7) == 0:
        return -1
    first7 = int(idx7[0])
    correct_until7 = int(np.sum(block_fb[: first7 + 1] == 1))
    return int(np.rint(correct_until7 / 7.0))


def segment_blocks(fb: np.ndarray, gs: np.ndarray, session: np.ndarray, verbose: bool = True) -> List[Block]:
    fb = fb.astype(int).copy()
    gs_i = _to_int_gs(gs)
    session = session.astype(int).copy()

    blocks: List[Block] = []
    for s in np.unique(session):
        idxs = np.where(session == s)[0]
        in_block = False
        cur = []

        for t in idxs:
            if not in_block:
                if gs_i[t] == 1:
                    in_block = True
                    cur = [int(t)]
                else:
                    continue
            else:
                cur.append(int(t))

            # Block ends on a CHECK with gs==7
            if in_block and (fb[t] == 0) and (gs_i[t] == 7):
                bfb = fb[cur]
                bgs = gs_i[cur]
                bch = (bfb != 0).astype(int)  # work=1, check=0

                ok = (bgs[0] == 1) and (bgs[-1] == 7) and (bfb[-1] == 0)
                if ok:
                    rate_raw = compute_progress_rate_label(bfb, bgs)
                    if rate_raw in (2, 3, 4, 5):
                        blocks.append(Block(
                            session_id=int(s),
                            fb=bfb,
                            gs=bgs,
                            choice=bch,
                            rate_raw=rate_raw,
                            rate_label=int(rate_raw - 1),  # 1..4
                            n_checks=int(np.sum(bfb == 0)),
                        ))

                in_block = False
                cur = []

    if verbose:
        rates = [b.rate_raw for b in blocks]
        print(f"[segment_blocks] kept {len(blocks)} blocks total")
        for r in (2, 3, 4, 5):
            print(f"  raw rate {r} (label {r-1}): {sum(rr == r for rr in rates)} blocks")
    return blocks


class BlockDataset(Dataset):
    def __init__(self, blocks: List[Block]):
        self.blocks = blocks

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        b = self.blocks[idx]
        return {
            "choice": b.choice.astype(np.int64),
            "fb": b.fb.astype(np.int64),
            "gs": b.gs.astype(np.int64),
            "rate": np.int64(b.rate_label),
        }


def pad_collate(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    lengths = [len(x["gs"]) for x in batch]
    T = max(lengths)

    def pad(arr, v):
        out = np.full((T,), v, dtype=arr.dtype)
        out[: len(arr)] = arr
        return out

    choice = np.stack([pad(x["choice"], 0) for x in batch], 0)
    fb = np.stack([pad(x["fb"], 0) for x in batch], 0)
    gs = np.stack([pad(x["gs"], 1) for x in batch], 0)
    rate = np.array([x["rate"] for x in batch], dtype=np.int64)

    valid = np.zeros((len(batch), T), dtype=np.float32)
    for i, L in enumerate(lengths):
        valid[i, :L] = 1.0

    check_mask = ((fb == 0).astype(np.float32) * valid)

    return {
        "choice": torch.from_numpy(choice),
        "fb": torch.from_numpy(fb),
        "gs": torch.from_numpy(gs),
        "rate": torch.from_numpy(rate),
        "valid_mask": torch.from_numpy(valid),
        "check_mask": torch.from_numpy(check_mask),
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }


# -----------------------------
# Model: obs-aware LSTM + incremental belief
# -----------------------------
class GaugeObsLSTM(nn.Module):
    """
    Per trial t:
      Inputs: choice[t], fb[t], gs[t]
        - gs channel carries true gs ONLY on check trials (fb==0), otherwise 0.
        - For lesion-MSE test ONLY, gs channel can be masked after M checks.

      Output: belief p(g_t) over gauge (start-of-trial) emitted BEFORE check observation (no leakage).

      Supervision: NLL on check trials only.
    """
    def __init__(self, hidden_size=128, choice_emb=8, fb_emb=8, gs_emb=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.K = 7

        self.choice_embed = nn.Embedding(2, choice_emb)
        self.fb_embed = nn.Embedding(3, fb_emb)
        self.gs_embed = nn.Embedding(8, gs_emb)  # 0=no obs, 1..7 observed gauge

        self.in_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.cell = nn.LSTMCell(choice_emb + fb_emb + gs_emb, hidden_size)

        self.inc_head = nn.Linear(hidden_size, 1)

    @torch.no_grad()
    def init_state(self, B, device):
        h = torch.zeros(B, self.hidden_size, device=device)
        c = torch.zeros(B, self.hidden_size, device=device)
        return h, c

    def forward(self,
                choice: torch.Tensor, fb: torch.Tensor, gs: torch.Tensor, valid_mask: torch.Tensor,
                lesion_mode: str = "none",
                lesion_unit_mask: Optional[torch.Tensor] = None,  # (H,) float {0,1}
                max_obs_checks: Optional[int] = None):
        """
        lesion_mode:
          - "none"
          - "unit_always": zero selected hidden dims for all valid timesteps

        max_obs_checks:
          - None: normal (gauge channel on every check)
          - int M: allow gs channel only for first M checks in each block
                   (used ONLY for lesion MSE evaluation as requested)
        """
        device = choice.device
        B, T = choice.shape

        h, c = self.init_state(B, device)

        # belief over gauge
        gauge = torch.zeros(B, self.K, device=device)
        gauge[:, 0] = 1.0  # force start at 1

        check_count = torch.zeros(B, device=device)

        use_units = lesion_unit_mask is not None and lesion_unit_mask.numel() > 0
        if use_units:
            lesion_unit_mask = lesion_unit_mask.view(1, -1).to(device)  # (1,H)

        probs_list = []
        hs_list = []
        pinc_list = []

        eps = 1e-8

        for t in range(T):
            vt = valid_mask[:, t].unsqueeze(1)

            # 1) emit prediction BEFORE incorporating check observation (no leakage)
            probs_list.append(gauge.unsqueeze(1))

            # track checks
            is_check_t = ((fb[:, t] == 0).float() * valid_mask[:, t])
            check_count = check_count + is_check_t

            # build gs observation channel
            gs_t = gs[:, t].clamp(1, 7)
            if max_obs_checks is None:
                gs_obs = torch.where(fb[:, t] == 0, gs_t, torch.zeros_like(gs_t))
            else:
                allow = (check_count <= float(max_obs_checks) + 1e-6).float()
                gs_obs = torch.where((fb[:, t] == 0) & (allow > 0.5), gs_t, torch.zeros_like(gs_t))

            x = torch.cat([
                self.choice_embed(choice[:, t]),
                self.fb_embed(fb[:, t]),
                self.gs_embed(gs_obs),
            ], dim=-1)
            x = self.in_drop(x)

            h_new, c_new = self.cell(x, (h, c))
            h = vt * h_new + (1.0 - vt) * h
            c = vt * c_new + (1.0 - vt) * c

            # always-on unit lesion (matches your current forward pattern)
            if use_units and lesion_mode == "unit_always":
                kill = vt * lesion_unit_mask
                h = h * (1.0 - kill)
                c = c * (1.0 - kill)

            # learned increment probability (no hard-coded ignore incorrect)
            p_inc = torch.sigmoid(self.inc_head(h)) * vt  # (B,1)
            pinc_list.append(p_inc.unsqueeze(1))

            # belief update via one-step shift
            stay = 1.0 - p_inc
            gauge_next = torch.zeros_like(gauge)
            gauge_next[:, 0] = gauge[:, 0] * stay.squeeze(1)
            for i in range(1, self.K - 1):
                gauge_next[:, i] = gauge[:, i] * stay.squeeze(1) + gauge[:, i - 1] * p_inc.squeeze(1)
            gauge_next[:, self.K - 1] = gauge[:, self.K - 1] + gauge[:, self.K - 2] * p_inc.squeeze(1)
            gauge = vt * gauge_next + (1.0 - vt) * gauge

            hs_list.append(h.unsqueeze(1))

        probs = torch.cat(probs_list, 1)                 # (B,T,K) prediction before obs
        hs = torch.cat(hs_list, 1)                       # (B,T,H)
        pincs = torch.cat(pinc_list, 1) if len(pinc_list) else None  # (B,T,1)
        log_probs = torch.log(probs.clamp_min(eps))      # (B,T,K)

        return {"probs_pred": probs, "log_probs_pred": log_probs, "hs": hs, "p_inc": pincs}


# -----------------------------
# Objective + penalties
# -----------------------------
def masked_nll(log_probs: torch.Tensor, gs: torch.Tensor, check_mask: torch.Tensor) -> torch.Tensor:
    B, T, K = log_probs.shape
    target = (gs.long() - 1).clamp(0, K - 1)
    lp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    nll = -lp
    return (nll * check_mask).sum() / check_mask.sum().clamp(min=1.0)


def smoothness_penalty(hs: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if hs.shape[1] < 2:
        return hs.new_tensor(0.0)
    dh = hs[:, 1:] - hs[:, :-1]
    vm = (valid_mask[:, 1:] * valid_mask[:, :-1]).unsqueeze(-1)
    return (dh.pow(2) * vm).sum() / (vm.sum().clamp(min=1.0) * hs.shape[-1])


def incorrect_update_penalty(p_inc: torch.Tensor, fb: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """
    Discourage increments on incorrect work trials (fb==2) without hard-coding.
    Penalize p_inc on those trials.
    """
    if p_inc is None:
        return fb.new_tensor(0.0, dtype=torch.float32)
    p = p_inc.squeeze(-1)  # (B,T)
    mask = (fb == 2).float() * valid_mask
    denom = mask.sum().clamp(min=1.0)
    return (p * mask).sum() / denom


@torch.no_grad()
def eval_objective(model, loader, device, lambda_smooth: float, lambda_incorrect: float):
    model.eval()
    tot = 0.0
    nb = 0
    for batch in loader:
        choice = batch["choice"].to(device)
        fb = batch["fb"].to(device)
        gs = batch["gs"].to(device)
        valid = batch["valid_mask"].to(device)
        check = batch["check_mask"].to(device)

        out = model(choice, fb, gs, valid, lesion_mode="none", max_obs_checks=None)
        nll = masked_nll(out["log_probs_pred"], gs, check)
        smooth = smoothness_penalty(out["hs"], valid)
        incbad = incorrect_update_penalty(out["p_inc"], fb, valid)
        loss = nll + lambda_smooth * smooth + lambda_incorrect * incbad

        tot += loss.item()
        nb += 1
    return tot / max(nb, 1)


def train_model(model, train_loader, val_loader, device, epochs, lr, weight_decay,
                grad_clip, lambda_smooth, lambda_incorrect, patience, min_delta):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        tr = 0.0
        nb = 0
        for batch in train_loader:
            choice = batch["choice"].to(device)
            fb = batch["fb"].to(device)
            gs = batch["gs"].to(device)
            valid = batch["valid_mask"].to(device)
            check = batch["check_mask"].to(device)

            out = model(choice, fb, gs, valid, lesion_mode="none", max_obs_checks=None)
            nll = masked_nll(out["log_probs_pred"], gs, check)
            smooth = smoothness_penalty(out["hs"], valid)
            incbad = incorrect_update_penalty(out["p_inc"], fb, valid)
            loss = nll + lambda_smooth * smooth + lambda_incorrect * incbad

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            tr += loss.item()
            nb += 1

        val = eval_objective(model, val_loader, device, lambda_smooth=lambda_smooth, lambda_incorrect=lambda_incorrect)
        tr = tr / max(nb, 1)
        print(f"[epoch {ep:02d}] train_obj={tr:.4f}  val_obj={val:.4f}")

        improved = (best - val) > min_delta
        if improved:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("[early stop]")
                break

    if best_state is not None:
        model.load_state_dict(best_state)


# -----------------------------
# Block forward (numpy)
# -----------------------------
@torch.no_grad()
def forward_block(model: nn.Module, b: Block, device: torch.device,
                  lesion_mask: Optional[np.ndarray] = None,
                  max_obs_checks: Optional[int] = None) -> Dict[str, np.ndarray]:
    model.eval()
    choice = torch.from_numpy(b.choice[None]).long().to(device)
    fb = torch.from_numpy(b.fb[None]).long().to(device)
    gs = torch.from_numpy(b.gs[None]).long().to(device)
    valid = torch.ones(1, choice.shape[1], device=device)

    if lesion_mask is None:
        out = model(choice, fb, gs, valid, lesion_mode="none", max_obs_checks=max_obs_checks)
    else:
        m = torch.from_numpy(lesion_mask.astype(np.float32)).to(device)
        out = model(choice, fb, gs, valid, lesion_mode="unit_always", lesion_unit_mask=m, max_obs_checks=max_obs_checks)

    probs = out["probs_pred"][0].detach().cpu().numpy()      # (T,K)
    hs = out["hs"][0].detach().cpu().numpy()                # (T,H)
    pred_map = probs.argmax(-1) + 1                         # (T,)
    return {"probs": probs, "hs": hs, "pred_map": pred_map}


# -----------------------------
# Rate unit selection: regression on ALL trials (train blocks only)
# -----------------------------
@torch.no_grad()
def extract_hidden_all_trials(model, blocks: List[Block], device,
                             lesion_mask: Optional[np.ndarray] = None) -> List[Dict]:
    """
    Per-trial hidden states for rate regression fitting (ALL trials: work+check).
    Uses hs[t] (post-update at trial t).
    """
    samples = []
    for bi, b in enumerate(blocks):
        out = forward_block(model, b, device, lesion_mask=lesion_mask, max_obs_checks=None)
        hs = out["hs"]
        for t in range(len(b.fb)):
            samples.append({
                "block_id": bi,
                "t": t,
                "h": hs[t].copy(),
                "rate": int(b.rate_label),   # 1..4
                "fb": int(b.fb[t]),
                "choice": int(b.choice[t]),
                "gs": int(b.gs[t]),
            })
    return samples


def fit_rate_ridge_importance(train_samples: List[Dict], alpha: float = 1.0) -> np.ndarray:
    """
    Ridge regression y=rate_label on standardized X=hidden.
    Importance per unit = |w_j|.
    """
    X = np.stack([s["h"] for s in train_samples], axis=0)
    y = np.array([s["rate"] for s in train_samples], dtype=np.float64)

    pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    pipe.fit(X, y)
    w = pipe.named_steps["ridge"].coef_.astype(np.float64)  # (H,)
    return np.abs(w)


def make_topk_mask(importance: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    H = importance.shape[0]
    idx = np.argsort(-importance)[: min(k, H)]
    mask = np.zeros((H,), dtype=np.float32)
    mask[idx] = 1.0
    return mask, idx


# -----------------------------
# Sanity check plot (baseline vs rate lesion)
# -----------------------------
@torch.no_grad()
def plot_block_prediction(model, block: Block, device,
                          lesion_mask: Optional[np.ndarray],
                          title: str = ""):
    base = forward_block(model, block, device, lesion_mask=None, max_obs_checks=None)
    les = forward_block(model, block, device, lesion_mask=lesion_mask, max_obs_checks=None) if lesion_mask is not None else None

    t = np.arange(len(block.gs))
    checks = np.where(block.fb == 0)[0]

    plt.figure(figsize=(12, 3.8))
    plt.step(t, block.gs, where="post", label="True gauge", alpha=0.70)
    plt.step(t, base["pred_map"], where="post", label="Pred (baseline)", alpha=0.95)
    if les is not None:
        plt.step(t, les["pred_map"], where="post", label="Pred (rate lesion)", alpha=0.95)

    plt.scatter(checks, block.gs[checks], s=55, marker="o", label="Check trials")
    plt.ylim(0.5, 7.5)
    plt.yticks(range(1, 8))
    plt.xlabel("Trial (within block)")
    plt.ylabel("Gauge size (start-of-trial)")
    plt.title(title or f"Val block (rate label={block.rate_label}, checks={block.n_checks})")
    plt.legend(frameon=True, ncol=4, loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Progress-rate decoding vs check number with k=0 baseline from hidden BEFORE first check
# -----------------------------
@torch.no_grad()
def extract_hidden_by_check_index(model: nn.Module, blocks: List[Block], device: torch.device,
                                 lesion_mask: Optional[np.ndarray] = None,
                                 max_obs_checks: Optional[int] = None) -> List[Dict]:
    """
    Returns decoding samples:
      - check_k = 0: hidden right BEFORE the first check (hs[t_first_check-1])
      - check_k = k>=1: hidden at the k-th check trial index (hs[t_check]) [post-check]
    """
    samples = []
    for bi, b in enumerate(blocks):
        out = forward_block(model, b, device, lesion_mask=lesion_mask, max_obs_checks=max_obs_checks)
        hs = out["hs"]

        check_idx = np.where(b.fb == 0)[0]
        if len(check_idx) == 0:
            continue

        t0 = int(check_idx[0])
        if t0 > 0:
            samples.append({"block_id": bi, "rate": b.rate_label, "check_k": 0, "h": hs[t0 - 1].copy()})

        for k, ti in enumerate(check_idx, start=1):
            samples.append({"block_id": bi, "rate": b.rate_label, "check_k": k, "h": hs[int(ti)].copy()})

    return samples


def _make_splits(y: np.ndarray, seed: int, max_splits: int = 5):
    if len(np.unique(y)) < 2:
        return None
    n_splits = min(max_splits, 5)
    if n_splits < 2:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros_like(y), y))


def decode_rate_by_check_number_fair(samples_pre: List[Dict], samples_post: List[Dict],
                                    max_k: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fair comparison: SAME blocks + SAME folds at each k, including k=0.
    Trains separate decoders for pre and post condition (baseline vs lesion),
    but uses identical splits for a fair comparison.
    """
    ks_out, acc_pre_out, acc_post_out = [], [], []

    for k in range(0, max_k + 1):
        by_pre = {s["block_id"]: s for s in samples_pre if s["check_k"] == k}
        by_post = {s["block_id"]: s for s in samples_post if s["check_k"] == k}
        common = sorted(set(by_pre.keys()) & set(by_post.keys()))
        if len(common) < 20:
            continue

        Xpre = np.stack([by_pre[bid]["h"] for bid in common], axis=0)
        Xpost = np.stack([by_post[bid]["h"] for bid in common], axis=0)
        y = np.array([by_pre[bid]["rate"] for bid in common], dtype=int)

        splits = _make_splits(y, seed=seed, max_splits=5)
        if splits is None:
            continue

        fold_pre, fold_post = [], []
        for tr, te in splits:
            clf_pre = make_pipeline(StandardScaler(),
                                    LogisticRegression(solver="lbfgs", multi_class="multinomial",
                                                       max_iter=5000, random_state=seed))
            clf_post = make_pipeline(StandardScaler(),
                                     LogisticRegression(solver="lbfgs", multi_class="multinomial",
                                                        max_iter=5000, random_state=seed))
            clf_pre.fit(Xpre[tr], y[tr])
            fold_pre.append(accuracy_score(y[te], clf_pre.predict(Xpre[te])))

            clf_post.fit(Xpost[tr], y[tr])
            fold_post.append(accuracy_score(y[te], clf_post.predict(Xpost[te])))

        ks_out.append(k)
        acc_pre_out.append(float(np.mean(fold_pre)))
        acc_post_out.append(float(np.mean(fold_post)))

    return np.array(ks_out), np.array(acc_pre_out), np.array(acc_post_out)


def plot_decoding_curve(ks: np.ndarray, acc_base: np.ndarray, acc_les: np.ndarray, title: str):
    plt.figure(figsize=(7.6, 5.0))
    plt.axvspan(-0.25, 1.25, alpha=0.10, label="pre → 1st check")

    plt.plot(ks, acc_base, marker="o", label="Baseline")
    plt.plot(ks, acc_les, marker="o", label="Rate lesion (top-k)")

    plt.ylim(0.20, 1.0)
    plt.xlabel("Check number within block")
    plt.ylabel("Progress-rate decoding accuracy")
    plt.xticks(ks, ["0 (pre)" if k == 0 else str(int(k)) for k in ks])

    plt.title(title)
    plt.grid(True)
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.show()


# -----------------------------
# PCA by rate (refit per condition)
# -----------------------------
@torch.no_grad()
def extract_hidden_at_last_check(model: nn.Module, blocks: List[Block], device: torch.device,
                                lesion_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    One point per block: hidden at last check trial (post-check).
    """
    Xs, ys = [], []
    for b in blocks:
        out = forward_block(model, b, device, lesion_mask=lesion_mask, max_obs_checks=None)
        hs = out["hs"]
        check_idx = np.where(b.fb == 0)[0]
        if len(check_idx) == 0:
            continue
        t_last = int(check_idx[-1])
        Xs.append(hs[t_last].copy())
        ys.append(int(b.rate_label))
    return np.stack(Xs, 0), np.array(ys, dtype=int)


def plot_pca_two_panels(model, val_blocks, device, lesion_mask, title="Hidden state PCA by progress rate"):
    X0, y0 = extract_hidden_at_last_check(model, val_blocks, device, lesion_mask=None)
    X1, y1 = extract_hidden_at_last_check(model, val_blocks, device, lesion_mask=lesion_mask)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))
    for ax, X, y, name in [(axes[0], X0, y0, "Baseline"), (axes[1], X1, y1, "Rate lesion")]:
        pca = PCA(n_components=2, random_state=0)
        Z = pca.fit_transform(X)
        ev = pca.explained_variance_ratio_ * 100.0
        for r in [1, 2, 3, 4]:
            m = (y == r)
            ax.scatter(Z[m, 0], Z[m, 1], s=40, alpha=0.85, label=f"rate {r}")
        ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
        ax.set_title(name)
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=True, loc="upper right", bbox_to_anchor=(0.995, 0.995))
    fig.suptitle(title + " (last check / block)", y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    plt.show()


# -----------------------------
# Cross-rate feedback decoding (4x4 matrix)
# -----------------------------
@torch.no_grad()
def extract_feedback_samples(model: nn.Module, blocks: List[Block], device: torch.device) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Collect per-trial samples for feedback decoding (work trials only: fb in {1,2}).
    For each rate label r in 1..4: return X, y where y=1 for correct, 0 for incorrect.
    """
    by_rate = {r: {"X": [], "y": []} for r in [1, 2, 3, 4]}
    for b in blocks:
        out = forward_block(model, b, device, lesion_mask=None, max_obs_checks=None)
        hs = out["hs"]  # (T,H)
        for t in range(len(b.fb)):
            if b.fb[t] == 1 or b.fb[t] == 2:
                by_rate[b.rate_label]["X"].append(hs[t].copy())
                by_rate[b.rate_label]["y"].append(1 if b.fb[t] == 1 else 0)

    out_dict = {}
    for r in [1, 2, 3, 4]:
        X = np.stack(by_rate[r]["X"], 0) if len(by_rate[r]["X"]) else np.zeros((0, model.hidden_size))
        y = np.array(by_rate[r]["y"], dtype=int) if len(by_rate[r]["y"]) else np.zeros((0,), dtype=int)
        out_dict[r] = (X, y)
    return out_dict


def cross_rate_feedback_decoding_matrix(model: nn.Module, blocks: List[Block], device: torch.device, seed: int) -> np.ndarray:
    """
    Train decoder on one rate, test on another. Accuracy matrix (train_rate x test_rate).
    """
    data = extract_feedback_samples(model, blocks, device)
    rates = [1, 2, 3, 4]
    M = np.full((4, 4), np.nan, dtype=float)

    for i, r_tr in enumerate(rates):
        Xtr, ytr = data[r_tr]
        if len(ytr) < 50 or len(np.unique(ytr)) < 2:
            continue

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(solver="lbfgs", max_iter=4000, random_state=seed),
        )
        clf.fit(Xtr, ytr)

        for j, r_te in enumerate(rates):
            Xte, yte = data[r_te]
            if len(yte) < 50 or len(np.unique(yte)) < 2:
                continue
            pred = clf.predict(Xte)
            M[i, j] = accuracy_score(yte, pred)

    return M


def plot_cross_rate_matrix(M: np.ndarray, title: str = "Cross-rate generalization of feedback decoding"):
    rates = ["1", "2", "3", "4"]
    plt.figure(figsize=(6.4, 5.4))
    im = plt.imshow(M, vmin=0.5, vmax=1.0, interpolation="nearest")
    plt.xticks(range(4), rates)
    plt.yticks(range(4), rates)
    plt.xlabel("Test rate label")
    plt.ylabel("Train rate label")
    plt.title(title)

    # annotate
    for i in range(4):
        for j in range(4):
            if np.isfinite(M[i, j]):
                plt.text(j, i, f"{100*M[i,j]:.0f}", ha="center", va="center", fontsize=11)

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.linspace(0.5, 1.0, 6)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(100*t)}" for t in ticks])
    cbar.set_label("Accuracy (%)")

    plt.tight_layout()
    plt.show()


# -----------------------------
# Feedback axis cosine similarity: dot clouds across block pairs (off-diagonal)
# -----------------------------
@torch.no_grad()
def feedback_axes_per_block(model: nn.Module, blocks: List[Block], device: torch.device,
                            min_trials_per_class: int = 3) -> Dict[int, List[np.ndarray]]:
    """
    For each block, compute a 'feedback axis' in hidden space:
      axis = mean(h | correct work) - mean(h | incorrect work)
    Only work trials (fb in {1,2}).
    Returns axes grouped by rate label (1..4).
    """
    axes_by_rate = {r: [] for r in [1, 2, 3, 4]}
    for b in blocks:
        out = forward_block(model, b, device, lesion_mask=None, max_obs_checks=None)
        hs = out["hs"]
        idx_pos = np.where(b.fb == 1)[0]
        idx_neg = np.where(b.fb == 2)[0]
        if len(idx_pos) < min_trials_per_class or len(idx_neg) < min_trials_per_class:
            continue
        vpos = hs[idx_pos].mean(axis=0)
        vneg = hs[idx_neg].mean(axis=0)
        axis = vpos - vneg
        n = np.linalg.norm(axis)
        if n < 1e-8:
            continue
        axis = axis / n
        axes_by_rate[b.rate_label].append(axis.astype(np.float64))
    return axes_by_rate


def plot_cosine_dotclouds(axes_by_rate: Dict[int, List[np.ndarray]],
                          title: str = "Cosine similarity of feedback axes across rates",
                          max_points_per_pair: int = 400,
                          seed: int = 0):
    """
    Dot clouds of cosine similarity between feedback axes from DIFFERENT rates.
    No legend. Mean of all off-diagonal pairwise cosines is put in the title.
    Different colors per rate-pair. Subsampled for display only.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    pair_to_vals: Dict[str, np.ndarray] = {}
    pair_means: Dict[str, float] = {}

    for a in [1, 2, 3, 4]:
        for b in [a + 1, 2, 3, 4]:
            if b <= a:
                continue
            A = axes_by_rate.get(a, [])
            B = axes_by_rate.get(b, [])
            if len(A) == 0 or len(B) == 0:
                continue

            A = np.stack(A, axis=0).astype(float)
            B = np.stack(B, axis=0).astype(float)

            # defensive normalization
            A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
            B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)

            M = A @ B.T
            key = f"{a}-{b}"
            pair_to_vals[key] = M.reshape(-1)
            pair_means[key] = float(M.mean())

    if len(pair_to_vals) == 0:
        print("[cosine] no valid block axes for dot clouds.")
        return

    pairs = sorted(pair_to_vals.keys())
    cmap = plt.get_cmap("tab10")
    colors = {p: cmap(i % 10) for i, p in enumerate(pairs)}
    rng = np.random.RandomState(seed)

    # global mean (over all off-diagonal pairs)
    all_vals = np.concatenate([pair_to_vals[p] for p in pairs], axis=0)
    mean_offdiag = float(all_vals.mean())

    fig, ax = plt.subplots(figsize=(9.2, 5.0))

    for i, p in enumerate(pairs):
        v_full = pair_to_vals[p]
        if len(v_full) > max_points_per_pair:
            idx = rng.choice(len(v_full), size=max_points_per_pair, replace=False)
            v = v_full[idx]
        else:
            v = v_full

        x = np.full(len(v), float(i))
        xj = x + (rng.rand(len(v)) - 0.5) * 0.30

        ax.scatter(xj, v, s=14, alpha=0.35, color=colors[p], linewidths=0)
        # mean bar per pair (computed on ALL values)
        m = pair_means[p]
        ax.plot([i - 0.20, i + 0.20], [m, m], linewidth=3.0, color=colors[p])

    ax.axhline(mean_offdiag, linestyle="--", linewidth=2.0, alpha=0.7)

    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs)
    ax.set_ylabel("Cosine similarity")
    ax.set_xlabel("Progress-rate pair (rate a vs rate b)")
    ax.set_title(f"{title}\nmean off-diagonal = {mean_offdiag:.3f}", pad=16)
    ax.grid(True)
    ax.set_ylim(0.0, 1.02)

    fig.tight_layout()
    plt.show()


# -----------------------------
# Lesion MSE vs #lesioned units (random median + 5–95%), with gs masked after 2 checks ONLY here
# -----------------------------
@torch.no_grad()
def mse_all_trials(model: nn.Module, blocks: List[Block], device: torch.device,
                   lesion_mask: Optional[np.ndarray],
                   max_obs_checks: Optional[int]) -> float:
    """
    MSE of discrete MAP prediction vs true gs across ALL trials, averaged equally over blocks.
    """
    mses = []
    for b in blocks:
        out = forward_block(model, b, device, lesion_mask=lesion_mask, max_obs_checks=max_obs_checks)
        pred = out["pred_map"].astype(np.float64)
        true = b.gs.astype(np.float64)
        mses.append(float(np.mean((pred - true) ** 2)))
    return float(np.mean(mses)) if len(mses) else float("nan")


def p_star(p: float) -> str:
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return ""


def lesion_mse_curve(model: nn.Module, val_blocks: List[Block], device: torch.device,
                     importance: np.ndarray, k_list: List[int],
                     n_random: int, seed: int,
                     max_obs_checks_mse: int = 2):
    rng = np.random.RandomState(seed)
    H = importance.shape[0]
    rank = np.argsort(-importance)

    baseline = mse_all_trials(model, val_blocks, device, lesion_mask=None, max_obs_checks=max_obs_checks_mse)

    results = {
        "k": [],
        "baseline": baseline,
        "topk": [],
        "rand_med": [],
        "rand_p05": [],
        "rand_p95": [],
        "p_top_worse_than_rand": [],
    }

    for k in k_list:
        k = int(k)
        results["k"].append(k)

        # top-k mask
        km = np.zeros((H,), dtype=np.float32)
        km[rank[: min(k, H)]] = 1.0
        top = mse_all_trials(model, val_blocks, device, lesion_mask=km, max_obs_checks=max_obs_checks_mse)
        results["topk"].append(top)

        # random masks
        rvals = []
        for _ in range(n_random):
            ridx = rng.choice(H, size=min(k, H), replace=False)
            rm = np.zeros((H,), dtype=np.float32)
            rm[ridx] = 1.0
            rvals.append(mse_all_trials(model, val_blocks, device, lesion_mask=rm, max_obs_checks=max_obs_checks_mse))
        rvals = np.array(rvals, dtype=float)

        results["rand_med"].append(float(np.median(rvals)))
        results["rand_p05"].append(float(np.percentile(rvals, 5)))
        results["rand_p95"].append(float(np.percentile(rvals, 95)))

        # one-sided p: top-k MSE > random MSE  (top should be in the upper tail)
        p = (np.sum(rvals >= top) + 1.0) / (len(rvals) + 1.0)
        results["p_top_worse_than_rand"].append(float(p))

    return results


def plot_lesion_mse_curve(res: dict, title: str):
    k = np.array(res["k"], dtype=int)
    top = np.array(res["topk"], dtype=float)
    rmed = np.array(res["rand_med"], dtype=float)
    rp05 = np.array(res["rand_p05"], dtype=float)
    rp95 = np.array(res["rand_p95"], dtype=float)
    pvals = np.array(res["p_top_worse_than_rand"], dtype=float)
    base = float(res["baseline"])

    fig, ax = plt.subplots(figsize=(8.9, 5.3))

    ax.axhline(base, linestyle="--", label="Baseline (no lesion)", zorder=1)

    ax.fill_between(k, rp05, rp95, alpha=0.18, label="Random lesion (5–95%)", zorder=2)
    ax.plot(k, rmed, marker="o", label="Random lesion (median)", zorder=3)
    ax.plot(k, top, marker="o", label="Top-k rate lesion", zorder=4)

    # headroom so stars never collide with title or get clipped
    y_all = np.r_[top, rp95, base]
    y_min = float(np.nanmin(y_all))
    y_max = float(np.nanmax(y_all))
    y_span = max(1e-6, y_max - y_min)
    ax.set_ylim(y_min - 0.08 * y_span, y_max + 0.30 * y_span)

    # stars: one-sided p(top-k MSE > random MSE)
    for kk, tm, p in zip(k, top, pvals):
        s = p_star(float(p))
        if s:
            ax.annotate(
                s, (kk, tm),
                xytext=(0, 14), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=21, fontweight="bold",
                zorder=10, clip_on=False
            )

    ax.set_xlabel("Number of lesioned units (k)")
    ax.set_ylabel("MSE (all trials; MAP gauge prediction)")
    ax.set_title(title, pad=18)
    ax.grid(True)
    ax.legend(frameon=True, loc="upper left")

    # avoid tight_layout clipping annotations near the top
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    plt.show()


# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat_path", type=str, default="gauge_data.mat")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=64)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--lambda_smooth", type=float, default=1e-3)
    p.add_argument("--lambda_incorrect", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min_delta", type=float, default=1e-3)

    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--n_sanity_plots", type=int, default=12)

    p.add_argument("--max_decode_k", type=int, default=7)

    p.add_argument("--k_list", type=str, default="6,12,19,26")
    p.add_argument("--n_random_lesions", type=int, default=2000)

    p.add_argument("--max_obs_checks_mse", type=int, default=2)  # for lesion MSE ONLY
    p.add_argument("--rate_ridge_alpha", type=float, default=1.0)

    args = p.parse_args()

    set_pub_style()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    data = loadmat(args.mat_path)
    fb = data["fb"].squeeze().astype(int)
    gs = data["gs"].squeeze().astype(float)
    session = data["session"].squeeze().astype(int)

    blocks = segment_blocks(fb, gs, session, verbose=True)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(blocks))
    blocks = [blocks[i] for i in perm]

    n_train = int(args.train_frac * len(blocks))
    train_blocks = blocks[:n_train]
    val_blocks = blocks[n_train:]
    print(f"[split] train={len(train_blocks)}  val={len(val_blocks)}")

    train_loader = DataLoader(BlockDataset(train_blocks), batch_size=args.batch_size,
                              shuffle=True, num_workers=0, collate_fn=pad_collate)
    val_loader = DataLoader(BlockDataset(val_blocks), batch_size=args.batch_size,
                            shuffle=False, num_workers=0, collate_fn=pad_collate)

    model = GaugeObsLSTM(hidden_size=args.hidden_size, dropout=args.dropout)
    print(model)

    train_model(model, train_loader, val_loader, device,
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                grad_clip=args.grad_clip, lambda_smooth=args.lambda_smooth,
                lambda_incorrect=args.lambda_incorrect,
                patience=args.patience, min_delta=args.min_delta)

    # ---- Rate unit selection by regression on ALL trials (train blocks only) ----
    train_samples = extract_hidden_all_trials(model, train_blocks, device, lesion_mask=None)
    importance = fit_rate_ridge_importance(train_samples, alpha=args.rate_ridge_alpha)
    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    k_list = sorted(list(set([k for k in k_list if k > 0])))
    k_for_plots = max(k_list) if len(k_list) else 16
    lesion_mask_top, lesion_idx = make_topk_mask(importance, k_for_plots)
    print(f"[rate lesion] using top-k={k_for_plots} units. top indices (first 10): {lesion_idx[:10].tolist()}")

    # ---- Sanity check plots (baseline vs rate lesion only) ----
    n_show = min(args.n_sanity_plots, len(val_blocks))
    if n_show > 0:
        idxs = np.linspace(0, len(val_blocks) - 1, n_show).astype(int)
        for i in idxs:
            b = val_blocks[i]
            plot_block_prediction(model, b, device, lesion_mask_top,
                                  title=f"Val block idx={i} (rate label={b.rate_label})")

    # ---- Progress-rate decoding vs checks (includes k=0 pre-first-check) ----
    samp_base = extract_hidden_by_check_index(model, val_blocks, device, lesion_mask=None, max_obs_checks=None)
    samp_les  = extract_hidden_by_check_index(model, val_blocks, device, lesion_mask=lesion_mask_top, max_obs_checks=None)

    ks, acc_base, acc_les = decode_rate_by_check_number_fair(samp_base, samp_les, max_k=args.max_decode_k, seed=args.seed)
    plot_decoding_curve(ks, acc_base, acc_les, title="Progress-rate decoding vs #checks (val blocks)")

    # ---- PCA geometry (refit per panel), baseline vs lesion ----
    plot_pca_two_panels(model, val_blocks, device, lesion_mask_top,
                        title="Hidden state geometry by progress rate")

    # ---- Cross-rate feedback decoding matrix (baseline network) ----
    M = cross_rate_feedback_decoding_matrix(model, val_blocks, device, seed=args.seed)
    plot_cross_rate_matrix(M, title="Cross-rate generalization of feedback decoding (hidden units)")

    # ---- Cosine similarity of feedback axes across rates: dot clouds + mean off-diagonal ----
    axes_by_rate = feedback_axes_per_block(model, val_blocks, device, min_trials_per_class=3)
    plot_cosine_dotclouds(axes_by_rate, title="Feedback-axis cosine similarity across progress rates")

    # ---- Lesion MSE curve vs #units (ONLY here: mask gs input after 2 checks) ----
    res = lesion_mse_curve(model, val_blocks, device,
                           importance=importance,
                           k_list=k_list,
                           n_random=args.n_random_lesions,
                           seed=args.seed,
                           max_obs_checks_mse=args.max_obs_checks_mse)

    for kk, tm, rm, pval in zip(res["k"], res["topk"], res["rand_med"], res["p_top_worse_than_rand"]):
        print(f"[lesion MSE] k={kk:3d}  baseline={res['baseline']:.4f}  top={tm:.4f}  rand_med={rm:.4f}  p(top>w rand)={pval:.4g}")

    plot_lesion_mse_curve(res, title=f"Rate-unit lesion impairs gauge prediction when gauge is hidden after {args.max_obs_checks_mse} checks")

    print("Done.")


if __name__ == "__main__":
    main()
