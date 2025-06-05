import logging
from typing import Iterable, Mapping, Optional

import cvxpy as cp
import numpy as np
import torch
from tqdm import tqdm

from krcps.config import Config
from krcps.memberships import Membership, get_membership
from krcps.procedures.rcps import _rcps
from krcps.procedures.utils import CalibrationProcedure, register_procedure
from krcps.uqs import UncertaintyQuantification

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


def _gamma_loss_fn(
    i: np.ndarray = None,
    offset: np.ndarray = None,
    q: float = None,
    _lambda: cp.Variable = None,
):
    i_lambda = i + 2 * _lambda
    inv_i_lambda = cp.multiply(cp.inv_pos(i_lambda), offset)
    loss = 2 * (1 + q) * inv_i_lambda - q
    loss = cp.pos(loss)
    return loss


class Pk:
    def __init__(
        self,
        ground_truth: torch.Tensor = None,
        uq: UncertaintyQuantification = None,
        membership: Membership = None,
        epsilon: float = None,
        lambda_max: float = None,
        prob_size: int = None,
    ):
        l, u = uq(0)

        if ground_truth.ndim > 3:
            ground_truth = ground_truth.flatten(0, -3)
            l = l.flatten(0, -3)
            u = u.flatten(0, -3)

        k, nk, m = membership(ground_truth, l, u)

        n = ground_truth.size(0)
        d = np.prod(ground_truth.size()[-2:])
        prob_nk = np.round(prob_size / d * nk).astype(int)
        logger.info(
            f"Initializing optimization problem with"
            f" prob_size={np.sum(prob_nk):,},"
            f" prob_nk={prob_nk}"
        )

        prob_coords = ([], [])
        prob_lambda = []
        for c, prob_nc in enumerate(prob_nk):
            c_coords = torch.nonzero(m[:, :, c] == 1, as_tuple=True)
            coord_idx = rng.choice(len(c_coords[0]), size=prob_nc, replace=False)
            for j, c_coord in enumerate(c_coords):
                prob_coords[j].extend(c_coord[coord_idx].tolist())
            prob_lambda.extend(prob_nc * [c])

        _lambda = cp.Variable(k)
        _lambda.value = 0.5 * np.ones(k)
        q = cp.Parameter(nonneg=True)

        c = (l + u) / 2
        i = u - l
        offset = torch.abs(ground_truth - c)

        prob_coords = tuple([torch.tensor(c) for c in prob_coords])
        r_hat = cp.sum(
            _gamma_loss_fn(
                i=i[:, prob_coords[0], prob_coords[1]].numpy(),
                offset=offset[:, prob_coords[0], prob_coords[1]].numpy(),
                q=q,
                _lambda=_lambda[[prob_lambda]],
            )
        ) / (n * np.sum(prob_nk))

        obj = cp.Minimize(cp.sum(cp.multiply(prob_nk, _lambda)))
        constraints = [_lambda >= 0, _lambda <= 0.5, r_hat <= epsilon]

        self.pk = cp.Problem(obj, constraints)
        self.q = q
        self._lambda = _lambda
        self.m = m

    def solve(self, gamma: Iterable[float], verbose: bool = False):
        logger.info(f"Solving optimization problem for {len(gamma)} values of gamma")

        sols = []
        for gamma in tqdm(gamma):
            logger.info(f"Solving problem with gamma={gamma:.2f}")
            self.q.value = gamma / (1 - gamma)
            self.pk.solve(verbose=verbose, warm_start=True)
            logger.info(
                f"Optimization problem solved with status: {self.pk.status.upper()}"
            )
            sols.append(
                (
                    gamma,
                    torch.tensor(self._lambda.value, dtype=torch.float),
                    self.pk.value,
                )
            )

        best_sol = sorted(sols, key=lambda x: x[-1])[0]
        logger.info(
            f"Best solution found with gamma={best_sol[0]:.2f},"
            f" lambda={best_sol[1]}, obj={best_sol[2]:,.2f}"
        )
        return best_sol[1]


@register_procedure("krcps")
class kRCPS(CalibrationProcedure):
    def __init__(self, config: Config):
        super().__init__(config)
        self.n_opt = config.n_opt
        self.membership = get_membership(config)
        self.prob_size = config.prob_size
        self.gamma = config.gamma

    def __call__(
        self,
        ground_truth: torch.Tensor,
        prediction: torch.Tensor,
        eta: Optional[torch.Tensor] = None,
    ):
        n = ground_truth.size(0)

        perm_idx = rng.permutation(n)
        opt_idx = perm_idx[: self.n_opt]
        cal_idx = perm_idx[self.n_opt :]

        opt_ground_truth, opt_prediction = (
            ground_truth[opt_idx],
            prediction[opt_idx],
        )
        cal_ground_truth, cal_prediction = (
            ground_truth[cal_idx],
            prediction[cal_idx],
        )

        opt_uq = self.uq(opt_prediction, **self.norm)
        cal_uq = self.uq(cal_prediction, **self.norm)

        pk = Pk(
            ground_truth=opt_ground_truth,
            uq=opt_uq,
            membership=self.membership,
            epsilon=self.epsilon,
            lambda_max=self.lambda_max,
            prob_size=self.prob_size,
        )
        _lambda_k = pk.solve(self.gamma, verbose=False)

        _lambda = torch.matmul(pk.m, _lambda_k) + self.lambda_max
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
