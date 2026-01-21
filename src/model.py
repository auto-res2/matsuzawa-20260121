from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torchvision import models


class ResNet18Wrapper(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats = self.backbone(x)
        logits = self.classifier(feats)
        if return_features:
            return logits, feats
        return logits


class DynamicSoftAugmentationLoss(nn.Module):
    def __init__(
        self,
        base_temperature: float = 1.0,
        pmin: float = 0.5,
        k: float = 2.0,
        feature_dim: int = 512,
        projector_dim: int = 64,
    ):
        super().__init__()
        self.base_temperature = base_temperature
        self.pmin = pmin
        self.k = k
        self.modulator = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )
        self.feature_projector = nn.Linear(feature_dim, projector_dim)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        logits_aug: torch.Tensor,
        visibility: torch.Tensor,
        features: torch.Tensor,
        features_aug: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_classes = logits.size()
        device = logits.device
        soft_targets = self.pmin + (1 - self.pmin) * visibility.pow(self.k)
        one_hot = F.one_hot(targets, num_classes=num_classes).float().to(device)
        soft_labels = one_hot * soft_targets.unsqueeze(1) + (1 - one_hot) * (
            1 - soft_targets
        ).unsqueeze(1)
        soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)
        loss_ce = -torch.sum(soft_labels * F.log_softmax(logits, dim=1), dim=1).mean()

        avg_vis = visibility.mean().unsqueeze(0).unsqueeze(0)
        dyn_params = self.modulator(avg_vis)
        dynamic_weight = torch.sigmoid(dyn_params[0, 0])
        temperature = self.base_temperature + F.softplus(dyn_params[0, 1])

        scaled_logits = logits / temperature
        scaled_logits_aug = logits_aug / temperature
        loss_output_consistency = F.kl_div(
            F.log_softmax(scaled_logits, dim=1),
            F.softmax(scaled_logits_aug, dim=1),
            reduction="batchmean",
        )
        proj_f = self.feature_projector(features)
        proj_f_aug = self.feature_projector(features_aug)
        loss_feature_consistency = F.mse_loss(proj_f, proj_f_aug)
        return loss_ce + dynamic_weight * (loss_output_consistency + 0.5 * loss_feature_consistency)


class FixedSoftAugmentationLoss(nn.Module):
    def __init__(
        self,
        consistency_weight: float = 1.0,
        temperature: float = 1.0,
        pmin: float = 0.5,
        k: float = 2.0,
    ):
        super().__init__()
        self.consistency_weight = consistency_weight
        self.temperature = temperature
        self.pmin = pmin
        self.k = k

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, logits_aug: torch.Tensor, **_) -> torch.Tensor:
        batch_size, num_classes = logits.size()
        device = logits.device
        soft_targets = self.pmin + (1 - self.pmin) * torch.ones(batch_size, device=device)
        one_hot = F.one_hot(targets, num_classes=num_classes).float().to(device)
        soft_labels = one_hot * soft_targets.unsqueeze(1) + (1 - one_hot) * (
            1 - soft_targets
        ).unsqueeze(1)
        soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)
        loss_ce = -torch.sum(soft_labels * F.log_softmax(logits, dim=1), dim=1).mean()

        scaled_logits = logits / self.temperature
        scaled_logits_aug = logits_aug / self.temperature
        loss_consistency = F.kl_div(
            F.log_softmax(scaled_logits, dim=1),
            F.softmax(scaled_logits_aug, dim=1),
            reduction="batchmean",
        )
        return loss_ce + self.consistency_weight * loss_consistency


def get_model_and_loss(cfg: DictConfig) -> Tuple[nn.Module, nn.Module]:
    num_classes = cfg.dataset.num_classes
    if cfg.model.name.lower() == "resnet18":
        model = ResNet18Wrapper(num_classes=num_classes, pretrained=cfg.model.pretrained)
        feature_dim = 512
    else:
        raise ValueError(f"Unsupported model: {cfg.model.name}")

    method = cfg.method.lower()
    if "dynamic" in method:
        params = cfg.training.additional_params
        criterion = DynamicSoftAugmentationLoss(
            base_temperature=1.0,
            pmin=params.soft_target_params.pmin,
            k=params.soft_target_params.k,
            feature_dim=feature_dim,
            projector_dim=params.feature_projector_output_dim,
        )
    else:
        params = cfg.training.additional_params
        criterion = FixedSoftAugmentationLoss(
            consistency_weight=params.constant_consistency_weight,
            temperature=params.temperature,
            pmin=params.soft_target_params.pmin,
            k=params.soft_target_params.k,
        )
    return model, criterion
