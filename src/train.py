import os
import math
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import wandb

from src.preprocess import get_datasets
from src.model import get_model_and_loss

# -----------------------------------------------------------------------------
#                              Utility helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set python, numpy and torch RNG for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_fn(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return preds.eq(targets).float().mean().item()


def expected_calibration_error(
    probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    if probs.numel() == 0:
        return 0.0
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    bin_boundaries = torch.linspace(0.0, 1.0, steps=n_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_conf_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_conf_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()

# -----------------------------------------------------------------------------
#                           Training & Validation
# -----------------------------------------------------------------------------

def _train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    cfg: DictConfig,
    global_step: int,
) -> Tuple[int, float]:
    """Train for exactly one epoch, returning updated global_step and last batch acc."""
    model.train()
    last_train_acc = 0.0
    for batch_idx, (img, img_aug, vis, labels) in enumerate(loader):
        if cfg.mode == "trial" and batch_idx >= 2:
            break

        if epoch == 0 and batch_idx == 0:
            assert img.shape == img_aug.shape, "Clean/Aug image shape mismatch"
            assert img.shape[0] == labels.shape[0], "Batch size mismatch with labels"

        img, img_aug, vis, labels = [
            t.to(device, non_blocking=True) for t in (img, img_aug, vis, labels)
        ]

        logits, feats = model(img, return_features=True)
        logits_aug, feats_aug = model(img_aug, return_features=True)

        loss = criterion(
            logits=logits,
            targets=labels,
            logits_aug=logits_aug,
            visibility=vis.squeeze(),
            features=feats,
            features_aug=feats_aug,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss encountered (epoch={epoch}, step={global_step}): {loss.item()}"
            )

        loss.backward()

        # Gradient integrity assertions
        grad_norm_sum = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm_sum += p.grad.abs().sum().item()
        assert grad_norm_sum > 0.0, "All gradients are zero – graph disconnected!"
        assert not math.isnan(grad_norm_sum) and not math.isinf(grad_norm_sum), "Bad gradients!"

        nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        last_train_acc = accuracy_fn(logits, labels)
        if cfg.wandb.mode != "disabled":
            wandb.log(
                {
                    "train_loss": loss.item(),
                    "train_acc": last_train_acc,
                    "epoch": epoch,
                    "step": global_step,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
        global_step += 1
    return global_step, last_train_acc


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    cfg: DictConfig,
    prefix: str = "val",
) -> Dict[str, Any]:
    model.eval()
    total_correct = 0
    total_samples = 0
    losses: List[float] = []
    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for batch_idx, (img, labels) in enumerate(loader):
            if cfg.mode == "trial" and batch_idx >= 2:
                break
            img = img.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits, _ = model(img, return_features=True)
            probs = torch.softmax(logits, dim=1)
            losses.append(F.cross_entropy(logits, labels).item())
            total_correct += logits.argmax(1).eq(labels).sum().item()
            total_samples += labels.size(0)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    mean_loss = float(np.mean(losses)) if losses else 0.0
    acc = total_correct / max(1, total_samples)
    probs_cat = torch.cat(all_probs, dim=0) if all_probs else torch.empty(0)
    labels_cat = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)
    ece = expected_calibration_error(probs_cat, labels_cat) if probs_cat.numel() else 0.0
    cm = (
        confusion_matrix(labels_cat.numpy(), probs_cat.argmax(1).numpy())
        if probs_cat.numel()
        else None
    )

    metrics = {
        f"{prefix}_loss": mean_loss,
        f"{prefix}_acc": acc,
        f"{prefix}_ece": ece,
        f"{prefix}_confusion_matrix": cm.tolist() if cm is not None else None,
    }

    if cfg.wandb.mode != "disabled":
        log_dict = {k: v for k, v in metrics.items() if v is not None}
        log_dict["epoch"] = epoch
        wandb.log(log_dict)

    return metrics

# -----------------------------------------------------------------------------
#                         Optuna – Hyper-parameter Search
# -----------------------------------------------------------------------------

def _suggest_from_cfg(trial: optuna.Trial, search_spaces: List[Dict[str, Any]]) -> Dict[str, Any]:
    suggestions: Dict[str, Any] = {}
    for space in search_spaces:
        name = space["param_name"]
        dist = space["distribution_type"].lower()
        if dist == "loguniform":
            suggestions[name] = trial.suggest_float(
                name, float(space["low"]), float(space["high"]), log=True
            )
        elif dist == "uniform":
            suggestions[name] = trial.suggest_float(
                name, float(space["low"]), float(space["high"]), log=False
            )
        elif dist == "categorical":
            suggestions[name] = trial.suggest_categorical(name, space["choices"])
        else:
            raise ValueError(f"Unsupported Optuna distribution type: {dist}")
    return suggestions


def _run_quick_eval(cfg: DictConfig, hyperparams: Dict[str, Any]) -> float:
    cfg = OmegaConf.clone(cfg)
    OmegaConf.set_struct(cfg, False)
    for k, v in hyperparams.items():
        if k in cfg.training:
            cfg.training[k] = v
        else:
            cfg.training.additional_params[k] = v
    cfg.mode = "trial"
    cfg.wandb.mode = "disabled"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, _ = get_datasets(cfg, subset_ratio=0.15)
    train_loader = DataLoader(
        train_ds, batch_size=min(64, cfg.training.batch_size), shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=min(64, cfg.training.batch_size), shuffle=False, num_workers=2
    )
    model, criterion = get_model_and_loss(cfg)
    model.to(device)
    criterion.to(device)

    optimiser = torch.optim.SGD(
        model.parameters(),
        lr=cfg.training.learning_rate,
        momentum=0.9,
        weight_decay=cfg.training.weight_decay,
    )

    global_step = 0
    for epoch in range(3):
        global_step, _ = _train_one_epoch(
            model,
            criterion,
            train_loader,
            optimiser,
            device,
            epoch,
            cfg,
            global_step,
        )
    val_metrics = _evaluate(model, val_loader, device, epoch=2, cfg=cfg)
    return 1.0 - val_metrics["val_acc"]

# -----------------------------------------------------------------------------
#                                Main Entrypoint
# -----------------------------------------------------------------------------


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):  # noqa: C901
    if cfg.run is None:
        raise ValueError("run=<run_id> must be supplied.")

    run_id: str = cfg.run
    run_yaml = (
        Path(__file__).resolve().parent.parent / "config" / "runs" / f"{run_id}.yaml"
    )
    if not run_yaml.exists():
        raise FileNotFoundError(f"Run-specific YAML not found: {run_yaml}")

    run_cfg = OmegaConf.load(run_yaml)
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, run_cfg)

    if cfg.mode not in {"trial", "full"}:
        raise ValueError("mode must be 'trial' or 'full'")
    if cfg.mode == "trial":
        cfg.training.epochs = 1
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
    else:
        cfg.wandb.mode = "online"

    for section in ("training", "dataset", "model"):
        assert hasattr(cfg, section), f"Missing required config section: {section}"

    set_seed(int(cfg.training.seed))
    results_root = Path(cfg.results_dir).expanduser().resolve()
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------- Optuna ---------------------------------- #
    if cfg.optuna.n_trials and int(cfg.optuna.n_trials) > 0:

        def objective(trial: optuna.Trial) -> float:
            suggestions = _suggest_from_cfg(trial, cfg.optuna.search_spaces)
            cfg_trial = OmegaConf.clone(cfg)
            return _run_quick_eval(cfg_trial, suggestions)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=int(cfg.optuna.n_trials), show_progress_bar=True)
        best_params = study.best_params
        with open(run_dir / "optuna_best_params.json", "w") as fp:
            json.dump(best_params, fp, indent=2)
        for k, v in best_params.items():
            if k in cfg.training:
                cfg.training[k] = v
            else:
                cfg.training.additional_params[k] = v

    # -------------------------- Data preparation -------------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, test_ds = get_datasets(cfg)

    def _build_loader(ds, shuffle: bool):
        return DataLoader(
            ds,
            batch_size=cfg.training.batch_size,
            shuffle=shuffle,
            num_workers=min(os.cpu_count(), 8),
            pin_memory=torch.cuda.is_available(),
            drop_last=shuffle,
        )

    train_loader = _build_loader(train_ds, shuffle=True)
    val_loader = _build_loader(val_ds, shuffle=False)
    test_loader = _build_loader(test_ds, shuffle=False)

    # -------------------------- Model & Optimiser -------------------------- #
    model, criterion = get_model_and_loss(cfg)
    model.to(device)
    criterion.to(device)

    assert (
        hasattr(model, "num_classes") and model.num_classes == cfg.dataset.num_classes
    ), "Model's num_classes attribute mismatch!"

    if cfg.training.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.training.learning_rate,
            momentum=cfg.training.momentum,
            weight_decay=cfg.training.weight_decay,
        )
    elif cfg.training.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimiser: {cfg.training.optimizer}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)

    # --------------------------- WandB init -------------------------------- #
    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=run_id,
            resume="allow",
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
        )
        print(f"[WandB] URL: {wandb.run.get_url()}")

    # ------------------------------ Training ------------------------------ #
    best_val_acc = 0.0
    global_step = 0
    last_train_acc = 0.0
    for epoch in range(cfg.training.epochs):
        global_step, last_train_acc = _train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            device,
            epoch,
            cfg,
            global_step,
        )
        val_metrics = _evaluate(model, val_loader, device, epoch, cfg, prefix="val")
        scheduler.step()
        if val_metrics["val_acc"] > best_val_acc:
            best_val_acc = val_metrics["val_acc"]
            torch.save(model.state_dict(), run_dir / "best.pt")
        if cfg.wandb.mode != "disabled":
            wandb.summary.update(
                {
                    "best_val_acc": best_val_acc,
                    "val_acc": val_metrics["val_acc"],
                    "val_ece": val_metrics["val_ece"],
                    "val_confusion_matrix": val_metrics["val_confusion_matrix"],
                }
            )

    # ------------------------- Final evaluation --------------------------- #
    test_metrics = _evaluate(model, test_loader, device, epoch=cfg.training.epochs, cfg=cfg, prefix="test")
    torch.save(model.state_dict(), run_dir / "last.pt")

    if cfg.wandb.mode != "disabled":
        wandb.summary.update(
            {
                "accuracy": best_val_acc,
                "train_acc_final": last_train_acc,
                "test_acc": test_metrics["test_acc"],
                "test_ece": test_metrics["test_ece"],
                "test_confusion_matrix": test_metrics["test_confusion_matrix"],
                "final_checkpoint": str(run_dir / "last.pt"),
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()
