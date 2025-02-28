from abc import abstractmethod
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from skimage.filters import threshold_multiotsu

from krcps.config import Config
from krcps.losses import get_loss


class Membership:
    def __init__(self, config: Config):
        self.k = config.k

    @abstractmethod
    def __call__(self, ground_truth, l, u):
        pass


memberships: Mapping[str, Membership] = {}


def register_membership(name: str):
    def register(cls: Membership):
        memberships[name] = cls
        return cls

    return register


def get_membership(config: Config) -> Membership:
    return memberships[config.membership](config)


class LossQuantile(Membership):
    def __init__(self, config: Config):
        super().__init__(config)
        self.loss = None
        self.k = config.k

    def _get_quantiles(self, loss):
        q = torch.quantile(loss.view(-1), torch.arange(0, 1, 1 / self.k)[1:]).unique()
        k = len(q) + 1

        m = torch.bucketize(loss, q, right=False).squeeze()
        m = F.one_hot(m, num_classes=k).float()
        nk = torch.sum(m.view(-1, k), dim=0).numpy()
        return k, nk, m

    def __call__(self, ground_truth: torch.Tensor, l: torch.Tensor, u: torch.Tensor):
        loss = self.loss(ground_truth, l, u)
        return self._get_quantiles(loss)


@register_membership(name="mse_loss_quantile")
class MSELossQuantile(LossQuantile):
    def __init__(self, config: Config):
        super().__init__(config)
        self.loss = get_loss("vector_mse")


@register_membership(name="01_loss_quantile")
class BinaryLossQuantile(LossQuantile):
    def __init__(self, config: Config):
        super().__init__(config)
        self.loss = get_loss("vector_01")


@register_membership(name="01_loss_otsu")
class BinaryLossOtsu(Membership):
    def __init__(self, config: Config):
        super().__init__(config)
        self.loss = get_loss("vector_01")

    def _get_groups(self, loss):
        t = threshold_multiotsu(loss.numpy(), classes=self.k)
        k = len(t) + 1

        m = (k - 1) * torch.ones_like(loss, dtype=torch.long)
        for i, _t in enumerate(reversed(t)):
            m[loss <= _t] = k - (i + 2)

        tcoords = []
        for _k in range(k):
            tcoords.append(torch.nonzero(m == _k, as_tuple=True))

        assert len(tcoords) == len(t) + 1 == k
        assert all([len(_t[0]) == len(_t[1]) for _t in tcoords])
        assert sum([len(_t[0]) for _t in tcoords]) == torch.numel(loss)

        nk = np.empty((k))
        m = torch.zeros(loss.size(-2), loss.size(-1), k)
        for _k, _t in enumerate(tcoords):
            nk[_k] = len(_t[0])
            m[_t[0], _t[1], _k] = 1
        return k, nk, m

    def __call__(self, ground_truth: torch.Tensor, l: torch.Tensor, u: torch.Tensor):
        loss = self.loss(ground_truth, l, u)
        return self._get_groups(loss)
