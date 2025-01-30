import re
from typing import Any, Literal
from pydantic import BaseModel

from math_verify import parse, verify

from openreward.rewards.reward import Reward, RewardConfig
from openreward.types.chat import Chat, parse_chat


class MathVerifyRewardConfig(RewardConfig):
    """Configuration for the MathVerifyReward class."""

    type: Literal["MathVerify"] = "MathVerify"

    extraction_regex: str | None = None

    valid_expression_reward: float = 0.0
    """A partial reward for a valid mathematical expression that is not the correct answer."""


class MathVerifyReward(Reward):
    """Ensures that the model generates outputs that are valid mathematical expressions."""

    def __init__(self, config: MathVerifyRewardConfig):
        """Initializes the MathVerifyReward class."""
        super().__init__(config)

        if config.extraction_regex is None:
            # Extract between the last set of <answer> and </answer> tags
            extraction_regex = r"<answer>(.*?)</answer>"

        self._regex = re.compile(extraction_regex)

    def __call__(self, chat: Any) -> float | None:
        """Returns a reward of 1 if the text is a valid mathematical expression, valid_expression_reward (default 0) if there is a valid mathematical expression that is not correct, and 0 if there is no valid expression returned. Returns None to pass if the last message in the chat is not from the assistant."""
        chat: Chat = parse_chat(chat)

        if len(chat.messages) < 2 or chat.messages[-1].role != "assistant":
            return None

        text = chat.messages[-1].content
        try:
            parsed = parse(text)
        except:
            return 0.0
        return 1.0 if verify(parsed) else self.config.valid_expression_reward
