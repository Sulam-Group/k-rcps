from typing import Any, Iterable, Mapping

from ml_collections import ConfigDict


class Config(ConfigDict):
    def __init__(self, config_dict: Mapping[str, Any] = {}):
        super().__init__()
        self.procedure: str = config_dict.get("procedure", None)
        self.loss: str = config_dict.get("loss", None)
        self.bound: str = config_dict.get("bound", None)

        self.uq: str = config_dict.get("uq", None)
        self.norm_min: float = config_dict.get("norm_min", 0.0)
        self.norm_max: float = config_dict.get("norm_max", 1.0)
        self.norm_eps: float = config_dict.get("norm_eps", None)

        self.epsilon: float = config_dict.get("epsilon", None)
        self.delta: float = config_dict.get("delta", None)
        self.n_cal: int = config_dict.get("n_cal", None)
        self.n_val: int = config_dict.get("n_val", None)

        self.lambda_max: float = config_dict.get("lambda_max", None)
        self.stepsize: float = config_dict.get("stepsize", None)

        self.n_opt: int = config_dict.get("n_opt", None)
        self.gamma: Iterable[float] = config_dict.get("gamma", None)

        # K-RCPS settings
        self.membership: str = config_dict.get("membership", None)
        self.k: int = config_dict.get("k", None)
        self.prob_size: int = config_dict.get("prob_size", None)

        # sem-RCPS settings
        self.min_support: int = config_dict.get("min_support", None)
        self.max_support: int = config_dict.get("max_support", None)
        self.sem_control: bool = config_dict.get("sem_control", None)


configs: Mapping[str, Config] = {}


def register_config(name: str):
    def register(cls: Config):
        configs[name] = cls
        return cls

    return register


def get_config(name: str) -> Config:
    return configs[name]()
