import re
from typing import Any, Literal
from pydantic import BaseModel
import sympy as sp

from openreward.rewards.reward import RewardConfig, Reward
from openreward.types.chat import Chat, parse_chat


class CountdownRewardConfig(RewardConfig):
    """Configuration for the CountdownReward class."""

    type: Literal["Countdown"] = "Countdown"

    extraction_regex: str | None = None

    valid_format_reward: float = 0.0


class CountdownGroundTruth(BaseModel):
    """Ground truth for the CountdownReward class. The target isn't actually a solution - it's the target number. The solution is the expression that evaluates to the target number."""

    target: int
    starting_numbers: list[int]


class CountdownReward(Reward):
    """Tests the model's ability to play the number game Countdown."""

    def __init__(self, config: CountdownRewardConfig):
        """Initializes the CountdownReward class."""
        super().__init__(config)

        if config.extraction_regex is None:
            # Extract between the last set of <answer> and </answer> tags
            extraction_regex = r"<answer>(.*?)</answer>"

        self._regex = re.compile(extraction_regex)

        # Regex to split the expression into numbers and operators
        self._split_regex = re.compile(r"([0-9]+|[+*/-])")

        self._disallowed_operators = set(["**", "//"])

    def __call__(self, chat: Any) -> float | None:
        """Returns a reward of 1 if the text is a correct Countdown expression that solves the problem, valid_format_reward (default 0) if there is a valid Countdown expression that is not correct, and 0 if there is no valid expression returned. Returns None to pass."""

        chat: Chat = parse_chat(chat)
        try:
            ground_truth: CountdownGroundTruth = CountdownGroundTruth.model_validate(
                chat.metadata.ground_truth
            )
        except:
            return None

        if len(chat.messages) < 2 or chat.messages[-1].role != "assistant":
            return None

        text = chat.messages[-1].content

        try:
            answer = self._regex.findall(text)[-1]
        except:
            return 0.0

        # Check that the expression does not contain disallowed operators
        if any(operator in answer for operator in self._disallowed_operators):
            return 0.0

        # Check that the expression is a valid Countdown expression composed of the starting numbers
        split = self._split_regex.findall(answer)
        numbers = [int(x) for x in split if x.isdigit()]
        operators = [x for x in split if x in "+-*/"]

        if (
            set(numbers) != set(ground_truth.starting_numbers)
            or len(numbers) != len(ground_truth.starting_numbers)
            or len(operators) != len(ground_truth.starting_numbers) - 1
        ):  # This probably isn't comprehensive enough, but it's a start
            return self.config.valid_format_reward

        # Check that the expression evaluates to the target number
        try:
            parsed = sp.sympify(answer)
        except:
            return 0.0

        return 1.0 if parsed == ground_truth.target else self.config.valid_format_reward
