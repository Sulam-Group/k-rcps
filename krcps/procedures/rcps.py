import logging
from typing import Any, Mapping, Optional

import torch
from tqdm import tqdm

from krcps.bounds import ConcentrationBound
from krcps.config import Config
from krcps.losses import IntervalLoss
from krcps.procedures.utils import CalibrationProcedure, register_procedure
from krcps.uqs import UncertaintyQuantification

logger = logging.getLogger(__name__)


def _rcps(
    ground_truth: torch.Tensor = None,
    uq_fn: UncertaintyQuantification = None,
    loss_fn: IntervalLoss = None,
    bound: ConcentrationBound = None,
    epsilon: float = None,
    delta: float = None,
    lambda_max: torch.Tensor = None,
    stepsize: float = None,
    eta: Optional[torch.Tensor] = None,
):
    n_rcps = ground_truth.size(0)

    _lambda = lambda_max
    if not eta:
        eta = torch.ones_like(_lambda)

    loss = loss_fn(ground_truth, *uq_fn(_lambda))
    ucb = bound(n_rcps, delta, loss)
    if ucb > epsilon:
        raise ValueError(f"Initial UCB {ucb} > epsilon {epsilon}, increase lambda_max")

    pbar = tqdm(total=epsilon)
    pbar.update(ucb)
    pold = ucb

    while ucb <= epsilon:
        pbar.update(ucb - pold)
        pold = ucb

        prev_lambda = _lambda.clone()
        if torch.all(prev_lambda == 0):
            break

        _lambda -= stepsize * eta
        _lambda = torch.clamp(_lambda, min=0)

        loss = loss_fn(ground_truth, *uq_fn(_lambda))
        ucb = bound(n_rcps, delta, loss)
    _lambda = prev_lambda

    pbar.update(epsilon - pold)
    pbar.close()
    return _lambda


@register_procedure("rcps")
class RPCS(CalibrationProcedure):
    def __init__(self, config: Config):
        super().__init__(config)

    def __call__(
        self,
        ground_truth: torch.Tensor,
        prediction: torch.Tensor,
        uq_dict: Mapping[str, Any] = {},
        eta: Optional[torch.Tensor] = None,
    ):
        logger.info(f"Running RCPS with n_cal={len(ground_truth)}")

        _lambda = torch.tensor(self.lambda_max)
        return _rcps(
            ground_truth=ground_truth,
            uq_fn=self.uq(prediction, **uq_dict, **self.norm),
            loss_fn=self.loss,
            bound=self.bound,
            epsilon=self.epsilon,
            delta=self.delta,
            lambda_max=_lambda,
            stepsize=self.stepsize,
            eta=eta,
        )
