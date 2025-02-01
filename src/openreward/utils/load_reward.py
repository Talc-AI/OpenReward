from openreward.rewards.reward import Reward, RewardConfig
from openreward.rewards.math_verify_reward import (
    MathVerifyReward,
)
from openreward.rewards.thinking_format_reward import (
    ThinkingFormatReward,
)

import openreward.rewards.composite_reward as composite_reward  # Can't import CompositeReward directly because of circular import


def create_reward(config: any) -> Reward:

    if isinstance(config, dict):
        config = RewardConfig(**config)

    """Creates a reward object based on the configuration."""
    if config.type == "MathVerify":
        return MathVerifyReward(config)
    elif config.type == "ThinkingFormat":
        return ThinkingFormatReward(config)
    elif config.type == "Composite":
        return composite_reward.CompositeReward(config)
    else:
        raise ValueError(f"Unsupported reward type: {config.type}")
