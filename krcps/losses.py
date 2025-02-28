from abc import abstractmethod
from typing import Mapping, Optional

import torch
import torch.nn.functional as F


class IntervalLoss:
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(
        self, target: torch.Tensor, l: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        pass


losses: Mapping[str, IntervalLoss] = {}


def register_loss(name: str):
    def register(cls: IntervalLoss):
        losses[name] = cls
        return cls

    return register


def get_loss(name: str, **kwargs) -> IntervalLoss:
    return losses[name](**kwargs)


@register_loss(name="vector_mse")
class VectorMSELoss(IntervalLoss):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        target: torch.Tensor,
        l: torch.Tensor,
        u: torch.Tensor,
        reduction="mean",
        dim=0,
    ):
        c = (l + u) / 2
        loss = torch.pow(target - c, 2)

        if reduction == "mean":
            return torch.mean(loss, dim=dim)
        elif reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction {reduction}")


@register_loss(name="vector_01")
class VectorO1Loss(IntervalLoss):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        target: torch.Tensor,
        l: torch.Tensor,
        u: torch.Tensor,
        segmentation: Optional[torch.Tensor] = None,
        reduction="mean",
        dim=0,
    ):
        if segmentation is not None:
            m = F.one_hot(segmentation.long()).float()
            l = m * l[..., None]
            u = m * u[..., None]
            target = m * target[..., None]

        loss = torch.where(torch.logical_and(target >= l, target <= u), 0.0, 1.0)

        if reduction == "mean":
            return torch.mean(loss, dim=dim)
        elif reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction {reduction}")


@register_loss(name="01")
class Interval01Loss(VectorO1Loss):
    def __init__(self):
        super().__init__()

    def __call__(
        self, target: torch.Tensor, l: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        loss = super().__call__(target, l, u, reduction="none")
        return torch.mean(loss)


@register_loss(name="sem_01")
class Sem01Loss(IntervalLoss):
    def __init__(self, segmentation: torch.Tensor = None, target: torch.Tensor = None):
        super().__init__()
        self.m = F.one_hot(segmentation.long()).float()
        self.target = self.m * target[..., None]

        self.norm = torch.sum(self.m.view(-1, self.m.size(-1)), dim=0)
        self.mask = self.norm == 0

    def __call__(
        self,
        l: torch.Tensor,
        u: torch.Tensor,
    ):
        l = self.m * l[..., None]
        u = self.m * u[..., None]

        loss = torch.where(
            torch.logical_and(self.target >= l, self.target <= u), 0.0, 1.0
        )
        loss = loss.view(-1, self.m.size(-1))
        loss = torch.sum(loss, dim=0) / (self.norm + 1e-08)

        mask = self.mask.float()
        return mask * -1 + (1 - mask) * loss
