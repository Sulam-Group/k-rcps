import logging
from typing import Any, Mapping, Optional

import cvxpy as cp
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from krcps.bounds import ConcentrationBound
from krcps.config import Config
from krcps.losses import get_loss
from krcps.procedures.krcps import Pk, _gamma_loss_fn
from krcps.procedures.rcps import _rcps
from krcps.procedures.utils import CalibrationProcedure, register_procedure
from krcps.uqs import UncertaintyQuantification

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


def _sem_rcps(
    ground_truth: torch.Tensor = None,
    segmentation: torch.Tensor = None,
    uq_fn: UncertaintyQuantification = None,
    bound: ConcentrationBound = None,
    epsilon: float = None,
    delta: float = None,
    lambda_max: torch.Tensor = None,
    stepsize: float = None,
    eta: Optional[torch.Tensor] = None,
):
    organ_loss_fn = get_loss("sem_01", segmentation=segmentation, target=ground_truth)
    n_rcps = ground_truth.size(0)

    _lambda = lambda_max
    if not eta:
        eta = torch.ones_like(_lambda)

    loss_mask = organ_loss_fn.mask
    eta[loss_mask] = 0.0
    _lambda[loss_mask] = 0.0

    sem_loss = organ_loss_fn(*uq_fn(_lambda))
    sem_ucb = [bound(n_rcps, delta, l) if l != -1 else 1.0 for l in sem_loss]
    ucb = min([u if eta[i] == 1.0 else 1.0 for i, u in enumerate(sem_ucb)])
    if ucb > epsilon:
        raise ValueError(f"Initial UCB {ucb} > epsilon {epsilon}, increase lambda_max")

    pbar = tqdm(total=epsilon)
    pbar.update(ucb)
    pold = ucb

    while ucb <= epsilon:
        pbar.update(ucb - pold)
        pold = ucb

        prev_lambda = _lambda.clone()
        if torch.all(_lambda == 0):
            break

        _lambda -= stepsize * eta
        _lambda = torch.clamp(_lambda, min=0)
        eta *= _lambda != 0

        sem_loss = organ_loss_fn(*uq_fn(_lambda))
        sem_ucb = [bound(n_rcps, delta, l) if l != -1 else 1.0 for l in sem_loss]
        ucb = min([u if eta[i] == 1.0 else 1.0 for i, u in enumerate(sem_ucb)])

        exceeded = torch.tensor([1 if u > epsilon else 0 for u in sem_ucb]).bool()
        _lambda[exceeded] = prev_lambda[exceeded]
        eta *= ~exceeded

    pbar.update(epsilon - pold)
    pbar.close()
    return _lambda


class semPk(Pk):
    def __init__(
        self,
        ground_truth: torch.Tensor = None,
        segmentation: torch.Tensor = None,
        uq: UncertaintyQuantification = None,
        epsilon: float = None,
        min_support: int = None,
        max_support: int = None,
    ):
        m: torch.Tensor = F.one_hot(segmentation.long())
        k = m.size(-1)
        nk = (m.view(-1, k).sum(dim=0) / m.size(0)).numpy()

        d = np.prod(ground_truth.shape[1:])
        prob_size = np.round(min_support / min(nk[nk > 0]) * d)
        prob_nk = np.round(prob_size / d * nk).astype(int)
        prob_nk = np.clip(prob_nk, None, max_support)
        logger.info(
            f"Initializing semantic-Pk problem with"
            f" prob_size={np.sum(prob_nk):,},"
            f" prob_nk={prob_nk}"
        )

        prob_coords = tuple([[] for _ in ground_truth.shape])
        prob_lambda = []
        for c, prob_nc in enumerate(prob_nk):
            for i, _m in enumerate(m):
                m_nc = min(prob_nc, torch.sum(_m[..., c]).long().item())
                if m_nc == 0:
                    continue
                c_coords = torch.nonzero(_m[..., c], as_tuple=True)
                coord_idx = rng.choice(len(c_coords[0]), size=m_nc, replace=False)
                prob_coords[0].extend(m_nc * [i])
                for j, c_coord in enumerate(c_coords):
                    prob_coords[j + 1].extend(c_coord[coord_idx].tolist())
                prob_lambda.extend(m_nc * [c])
        logger.info(
            f"Number of selected pixels: {len(prob_coords[0]):,}/{ground_truth.numel():,}"
            f" ({len(prob_coords[0]) / ground_truth.numel():.2f}%)"
        )

        _lambda = cp.Variable(k)
        _lambda.value = 0.5 * np.ones(k)
        q = cp.Parameter(nonneg=True)

        l, u = uq(torch.zeros(k))
        c = (l + u) / 2
        i = u - l
        offset = torch.abs(ground_truth - c)

        prob_coords = tuple([torch.tensor(c) for c in prob_coords])
        r_hat = cp.sum(
            _gamma_loss_fn(
                i=i[prob_coords].numpy(),
                offset=offset[prob_coords].numpy(),
                q=q,
                _lambda=_lambda[[prob_lambda]],
            )
        ) / len(prob_coords[0])

        obj = cp.Minimize(cp.sum(cp.multiply(prob_nk, _lambda)))
        constraints = [_lambda >= 0, _lambda <= 0.5, r_hat <= epsilon]

        self.pk = cp.Problem(obj, constraints)
        self.q = q
        self._lambda = _lambda
        self.m = m


@register_procedure("semrcps")
class semRCPS(CalibrationProcedure):
    def __init__(self, config: Config):
        super().__init__(config)
        self.n_opt = config.n_opt
        self.min_support = config.min_support
        self.max_support = config.max_support
        self.gamma = config.gamma
        self.sem_control = config.sem_control

    def __call__(
        self,
        ground_truth: torch.Tensor,
        reconstruction: torch.Tensor,
        segmentation: torch.Tensor,
        uq_dict: Mapping[str, Any] = {},
        eta: Optional[torch.Tensor] = None,
    ):
        n = ground_truth.size(0)

        perm_idx = rng.permutation(n)
        opt_idx = perm_idx[: self.n_opt]
        cal_idx = perm_idx[self.n_opt :]

        logger.info(
            f"Running sem-RCPS with n_cal={len(cal_idx)},"
            f" n_opt={len(opt_idx)},"
            f" min_support={self.min_support}"
        )

        opt_ground_truth, opt_reconstruction, opt_segmentation = (
            ground_truth[opt_idx],
            reconstruction[opt_idx],
            segmentation[opt_idx],
        )
        cal_ground_truth, cal_reconstruction, cal_segmentation = (
            ground_truth[cal_idx],
            reconstruction[cal_idx],
            segmentation[cal_idx],
        )

        opt_uq = self.uq(opt_reconstruction, opt_segmentation, **uq_dict, **self.norm)
        cal_uq = self.uq(cal_reconstruction, cal_segmentation, **uq_dict, **self.norm)

        pk = semPk(
            ground_truth=opt_ground_truth,
            segmentation=opt_segmentation,
            uq=opt_uq,
            epsilon=self.epsilon,
            min_support=self.min_support,
            max_support=self.max_support,
        )
        _lambda_seg = pk.solve(self.gamma, verbose=False)

        _lambda = _lambda_seg + self.lambda_max
        if self.sem_control:
            return _sem_rcps(
                ground_truth=cal_ground_truth,
                segmentation=cal_segmentation,
                uq_fn=cal_uq,
                bound=self.bound,
                epsilon=self.epsilon,
                delta=self.delta,
                lambda_max=_lambda,
                stepsize=self.stepsize,
            )
        else:
            return _rcps(
                ground_truth=cal_ground_truth,
                uq_fn=cal_uq,
                loss_fn=self.loss,
                bound=self.bound,
                epsilon=self.epsilon,
                delta=self.delta,
                lambda_max=_lambda,
                stepsize=self.stepsize,
                eta=eta,
            )
