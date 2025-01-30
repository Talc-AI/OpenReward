import re
from typing import Any, Literal
from pydantic import BaseModel

from openreward.rewards.reward import RewardConfig, Reward
from openreward.types.chat import Chat, parse_chat


class ThinkingFormatRewardConfig(RewardConfig):
    """Configuration for the ThinkingFormatReward class."""

    type: Literal["ThinkingFormat"] = "ThinkingFormat"

    override_regex: str | None = None


class ThinkingFormatReward(Reward):
    """Ensures that the model generates outputs that conform to the <thinking></thinking><answer></answer> format."""

    def __init__(self, config: ThinkingFormatRewardConfig):
        """Initializes the ThinkingFormatReward class. Defaults to the format <thinking></thinking><answer></answer>. override_regex is a regular expression that can be used to override the default thinking format with a custom format."""
        super().__init__(config)

        if config.override_regex is None:
            override_regex = r"<thinking>(.*?)</thinking><answer>(.*?)</answer>"
        else:
            override_regex = config.override_regex

        self._regex = re.compile(override_regex)

    def __call__(self, chat: Any) -> float | None:
        """Returns a reward of 1 if the text matches the thinking format, and 0 otherwise. Returns None to pass if the last message in the chat is not from the assistant."""
        chat: Chat = parse_chat(chat)

        if len(chat.messages) < 2 or chat.messages[-1].role != "assistant":
            return None

        text = chat.messages[-1].content

        return 1.0 if self._regex.match(text) else 0.0
