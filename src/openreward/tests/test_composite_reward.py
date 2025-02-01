import unittest
from openreward.rewards.composite_reward import CompositeReward, CompositeRewardConfig
from openreward.rewards.math_verify_reward import MathVerifyRewardConfig
from openreward.rewards.thinking_format_reward import ThinkingFormatRewardConfig
from openreward.types.chat import Chat, ChatCompletion


class TestCompositeReward(unittest.TestCase):

    def setUp(self):
        self.config = CompositeRewardConfig(
            mode="sum",
            rewards=[
                MathVerifyRewardConfig(valid_expression_reward=0.5),
                ThinkingFormatRewardConfig(),
            ],
        )
        self.reward = CompositeReward(self.config)

    def test_sum_mode(self):
        chat = Chat(
            messages=[
                ChatCompletion(role="user", content="Solve the math problem 2 + 3."),
                ChatCompletion(
                    role="assistant",
                    content="<thinking>2 + 3 = 5</thinking><answer>5</answer>",
                ),
            ],
            metadata={"ground_truth": 5},
        )
        self.assertEqual(self.reward(chat), 1.5)

    def test_average_mode(self):
        self.config.mode = "average"
        self.reward = CompositeReward(self.config)
        chat = Chat(
            messages=[
                ChatCompletion(role="user", content="Solve the math problem 2 + 3."),
                ChatCompletion(
                    role="assistant",
                    content="<thinking>2 + 3 = 5</thinking><answer>5</answer>",
                ),
            ],
            metadata={"ground_truth": 5},
        )
        self.assertEqual(self.reward(chat), 0.75)

    def test_min_mode(self):
        self.config.mode = "min"
        self.reward = CompositeReward(self.config)
        chat = Chat(
            messages=[
                ChatCompletion(role="user", content="Solve the math problem 2 + 3."),
                ChatCompletion(
                    role="assistant",
                    content="<thinking>2 + 3 = 5</thinking><answer>5</answer>",
                ),
            ],
            metadata={"ground_truth": 5},
        )
        self.assertEqual(self.reward(chat), 0.5)

    def test_max_mode(self):
        self.config.mode = "max"
        self.reward = CompositeReward(self.config)
        chat = Chat(
            messages=[
                ChatCompletion(role="user", content="Solve the math problem 2 + 3."),
                ChatCompletion(
                    role="assistant",
                    content="<thinking>2 + 3 = 5</thinking><answer>5</answer>",
                ),
            ],
            metadata={"ground_truth": 5},
        )
        self.assertEqual(self.reward(chat), 1.0)


if __name__ == "__main__":
    unittest.main()
