from abc import ABC, abstractmethod
from typing import Any, Literal, Union
from pydantic import BaseModel


class RewardConfig(BaseModel):
    """Base configuration for reward classes."""

    type: str


class Reward(ABC):
    """Abstract base class for rewards."""

    def __init__(self, config: RewardConfig):
        self.config = config

    @abstractmethod
    def __call__(self, chat: Any) -> float | None:
        pass
