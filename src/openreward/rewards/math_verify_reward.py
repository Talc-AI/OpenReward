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
            # Extract between the last set of <answer> and </answer> tags, allowing for newlines in the answer
            extraction_regex = r"<answer>([\s\S]*?)</answer>"
        else:
            extraction_regex = config.extraction_regex

        self._regex = re.compile(extraction_regex)

    def __call__(self, chat: Any) -> float | None:
        """Returns a reward of 1 if the text is a valid mathematical expression, valid_expression_reward (default 0) if there is a valid mathematical expression that is not correct, and 0 if there is no valid expression returned. Returns None to pass if the last message in the chat is not from the assistant."""
        chat: Chat = parse_chat(chat)

        ground_truth = chat.metadata.ground_truth

        if (
            len(chat.messages) < 2
            or chat.messages[-1].role != "assistant"
            or not ground_truth
        ):
            return None

        try:
            text = chat.messages[-1].content
            answer = self._regex.findall(text)[-1]
            parsed = parse(answer)
            ground_truth = parse(str(ground_truth))
        except Exception as e:
            return 0.0

        if len(ground_truth) == 0:
            return None

        if len(parsed) == 0:
            return 0.0

        return (
            1.0 if verify(ground_truth, parsed) else self.config.valid_expression_reward
        )
