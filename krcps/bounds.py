from abc import abstractmethod
from typing import Mapping

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom


class ConcentrationBound:
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, n: int, delta: float, loss: float) -> float:
        pass


bounds: Mapping[str, ConcentrationBound] = {}


def register_bound(name: str):
    def register(cls: ConcentrationBound):
        bounds[name] = cls
        return cls

    return register


def get_bound(name: str) -> ConcentrationBound:
    return bounds[name]()


@register_bound(name="hoeffding")
class HoeffdingBound(ConcentrationBound):
    def __init__(self):
        super().__init__()

    def __call__(self, n: int, delta: float, loss: float) -> float:
        return (loss + np.sqrt(1 / (2 * n) * np.log(1 / delta))).item()


@register_bound(name="hoeffding_bentkus")
class HoeffdingBentkusBound(ConcentrationBound):
    def __init__(self, maxiter=1000):
        self.maxiter = maxiter

    def __call__(self, n: int, delta: float, loss: float) -> float:
        loss = loss + 1e-06

        def _hoeffding_plus(r, loss, n):
            h1 = lambda u: u * np.log(u / r) + (1 - u) * np.log((1 - u) / (1 - r))
            return -n * h1(np.maximum(r, loss))

        def _bentkus_plus(r, loss, n):
            return np.log(np.maximum(binom.cdf(np.floor(n * loss), n, r), 1e-10)) + 1

        def _tailprob(r):
            hoeffding_mu = _hoeffding_plus(r, loss, n)
            bentkus_mu = _bentkus_plus(r, loss, n)
            return np.minimum(hoeffding_mu, bentkus_mu) - np.log(delta)

        if _tailprob(1 - 1e-10) > 0:
            return 1.0
        else:
            try:
                return brentq(_tailprob, loss, 1 - 1e-10, maxiter=self.maxiter)
            except Exception as e:
                print(f"BRENTQ RUNTIME ERROR at muhat={loss}")
                print(e)
                return 1.0


@register_bound(name="crc")
class ConformalRiskControl(ConcentrationBound):
    def __init__(self, b: float = 1.0):
        self.b = b

    def __call__(self, n: int, delta: float, loss: float):
        return (n / (n + 1) * loss + self.b / (n + 1)).item()
