from typing import List, Any, Literal
import openreward.utils.load_reward as load_reward
from openreward.rewards.reward import Reward, RewardConfig


class CompositeRewardConfig(RewardConfig):
    """Configuration for the CompositeReward class."""

    type: Literal["Composite"] = "Composite"

    mode: Literal["sum", "average", "min", "max"] = "sum"
    rewards: List[RewardConfig]


class CompositeReward(Reward):
    """Combines multiple reward functions into a single score based on the specified mode."""

    def __init__(self, config: CompositeRewardConfig):
        """Initializes the CompositeReward class."""
        super().__init__(config)
        self.rewards = [load_reward.create_reward(reward) for reward in config.rewards]
        self.mode = config.mode

    def __call__(self, chat: Any) -> float | None:
        """Calls all reward functions and combines their scores based on the specified mode."""
        scores = [reward(chat) for reward in self.rewards]
        scores = [score for score in scores if score is not None]

        if not scores:
            return None

        if self.mode == "sum":
            return sum(scores)
        elif self.mode == "average":
            return sum(scores) / len(scores)
        elif self.mode == "min":
            return min(scores)
        elif self.mode == "max":
            return max(scores)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
