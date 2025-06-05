from abc import abstractmethod
from typing import Mapping, Optional, Tuple

import torch
import torch.nn.functional as F


def _normalize(
    l: torch.Tensor,
    u: torch.Tensor,
    min: float = 0.0,
    max: float = 1.0,
    eps: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    l, u = torch.clamp(l, min=min, max=max), torch.clamp(u, min=min, max=max)
    if eps is not None:
        l[l <= eps] = 0.0
        u[u <= eps] = 0.0
    return l, u


def _calibrated_quantile(
    sampled: torch.Tensor, alpha: float, dim: Optional[int] = None
):
    m = sampled.size(dim)
    q = torch.quantile(
        sampled,
        torch.tensor(
            [
                torch.floor(torch.tensor((m + 1) * alpha / 2)) / m,
                torch.min(
                    torch.tensor(
                        [1, torch.ceil(torch.tensor((m + 1) * (1 - alpha / 2))) / m]
                    )
                ),
            ]
        ),
        dim=dim,
    )
    l, u = q[0], q[1]
    return l, u


class UncertaintyQuantification:
    def __init__(
        self,
        min: float = 0.0,
        max: float = 1.0,
        eps: Optional[float] = None,
        segmentation: Optional[torch.Tensor] = None,
        num_classes: int = -1,
    ):
        self.min = min
        self.max = max
        self.eps = eps
        self.m = None
        if segmentation is not None:
            self.m = F.one_hot(segmentation.long(), num_classes=num_classes).float()

    @abstractmethod
    def _I(self, _lambda: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def __call__(self, _lambda: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.m is not None:
            _lambda = torch.matmul(self.m, _lambda)
        return _normalize(
            *self._I(_lambda),
            min=self.min,
            max=self.max,
            eps=self.eps,
        )


uqs: Mapping[str, UncertaintyQuantification] = {}


def register_uq(name: str):
    def register(cls: UncertaintyQuantification):
        uqs[name] = cls
        return cls

    return register


def get_uq(name: str):
    return uqs[name]


@register_uq(name="mse")
class MSE(UncertaintyQuantification):
    def __init__(self, denoised: torch.Tensor, **uq_kwargs):
        super().__init__(**uq_kwargs)
        self.mse = denoised[:, 1]

    def _I(self, _lambda: torch.Tensor):
        l_lambda = self.mse - _lambda
        u_lambda = self.mse + _lambda
        return l_lambda, u_lambda


@register_uq(name="qr_additive")
class AdditiveQuantileRegression(UncertaintyQuantification):
    def __init__(self, denoised: torch.Tensor, **uq_kwargs):
        super().__init__(**uq_kwargs)
        _l, _u = denoised[:, 0], denoised[:, 2]
        l, u = torch.minimum(_l, _u), torch.maximum(_l, _u)
        assert torch.all(u - l >= 0)
        self.l, self.u = l, u

    def _I(self, _lambda: torch.Tensor):
        l_lambda = self.l - _lambda
        u_lambda = self.u + _lambda
        return l_lambda, u_lambda


@register_uq(name="qr_multiplicative")
class MultiplicativeQuantileRegression(UncertaintyQuantification):
    def __init__(self, denoised: torch.Tensor, **uq_kwargs):
        super().__init__(**uq_kwargs)
        _l, x, _u = denoised[:, 0], denoised[:, 1], denoised[:, 2]

        q_eps = 1e-02
        l, u = torch.clamp(_l, min=q_eps), torch.clamp(_u, min=q_eps)

        self.l, self.x, self.u = l, x, u

    def _I(self, _lambda: torch.Tensor):
        l_lambda = self.x - _lambda * self.l
        u_lambda = self.x + _lambda * self.l
        return l_lambda, u_lambda


@register_uq(name="std")
class STD(UncertaintyQuantification):
    def __init__(self, sampled: torch.Tensor, dim: Optional[int] = None, **uq_kwargs):
        super().__init__(**uq_kwargs)
        mu, _std = torch.mean(sampled, dim=dim), torch.std(sampled, dim=dim)

        std_min = 1e-02
        std = torch.clamp(_std, min=std_min)

        self.mu, self.std = mu, std

    def _I(self, _lambda: torch.Tensor):
        l_lambda = self.mu - _lambda * self.std
        u_lambda = self.mu + _lambda * self.std
        return l_lambda, u_lambda


@register_uq(name="naive_sampling_additive")
class NaiveSamplingAdditive(UncertaintyQuantification):
    def __init__(
        self,
        sampled: torch.Tensor,
        alpha: float = None,
        dim: Optional[int] = None,
        **uq_kwargs,
    ):
        super().__init__(**uq_kwargs)
        q = torch.quantile(sampled, torch.tensor([alpha / 2, 1 - alpha / 2]), dim=dim)
        self.l, self.u = q[0], q[1]

    def _I(self, _lambda: torch.Tensor):
        l_lambda = self.l - _lambda
        u_lambda = self.u + _lambda
        return l_lambda, u_lambda


@register_uq(name="calibrated_quantile")
class CalibratedQuantile(UncertaintyQuantification):
    def __init__(
        self,
        sampled: torch.Tensor,
        alpha: float = None,
        dim: Optional[int] = None,
        **uq_kwargs,
    ):
        super().__init__(**uq_kwargs)
        l, u = _calibrated_quantile(sampled, alpha, dim=dim)
        self.l, self.u = l, u

    def _I(self, _lambda: torch.Tensor):
        l_lambda = self.l - _lambda
        u_lambda = self.u + _lambda
        return l_lambda, u_lambda


@register_uq(name="conffusion_multiplicative")
class MultiplicativeConffusion(UncertaintyQuantification):
    def __init__(self, denoised: torch.Tensor, **uq_kwargs):
        super().__init__(**uq_kwargs)
        _l, _u = denoised[:, 0], denoised[:, 2]
        l, u = torch.minimum(_l, _u), torch.maximum(_l, _u)
        assert torch.all(u - l >= 0)
        self.l, self.u = l, u

    def _I(self, _lambda: torch.Tensor):
        l_lambda = self.l / _lambda
        u_lambda = self.u * _lambda
        return l_lambda, u_lambda


@register_uq(name="conffusion_additive")
class AdditiveConffusion(UncertaintyQuantification):
    def __init__(self, denoised: torch.Tensor, **uq_kwargs):
        super().__init__(**uq_kwargs)
        _l, _u = denoised[:, 0], denoised[:, 2]
        l, u = torch.minimum(_l, _u), torch.maximum(_l, _u)
        assert torch.all(u - l >= 0)
        self.l, self.u = l, u

    def _I(self, _lambda: torch.Tensor):
        l_lambda = self.l - _lambda
        u_lambda = self.u + _lambda
        return l_lambda, u_lambda
