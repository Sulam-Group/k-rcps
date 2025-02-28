from abc import abstractmethod
from typing import Any, Mapping, Optional

import torch

from krcps.bounds import get_bound
from krcps.config import Config
from krcps.losses import get_loss
from krcps.uqs import get_uq


class CalibrationProcedure:
    def __init__(self, config: Config):
        self.uq = get_uq(config.uq)
        self.loss = get_loss(config.loss)
        self.bound = get_bound(config.bound)

        self.epsilon = config.epsilon
        self.delta = config.delta
        self.lambda_max = config.lambda_max
        self.stepsize = config.stepsize

        self.norm = {
            "norm_min": config.norm_min,
            "norm_max": config.norm_max,
            "norm_eps": config.norm_eps,
        }

    @abstractmethod
    def __call__(
        self,
        ground_truth: torch.Tensor,
        prediction: torch.Tensor,
        uq_dict: Mapping[str, Any] = {},
        eta: Optional[torch.Tensor] = None,
    ):
        pass


procedures: Mapping[str, CalibrationProcedure] = {}


def register_procedure(name: str):
    def register(cls: CalibrationProcedure):
        procedures[name] = cls
        return cls

    return register


def get_procedure(config: Config) -> CalibrationProcedure:
    return procedures[config.procedure](config)
