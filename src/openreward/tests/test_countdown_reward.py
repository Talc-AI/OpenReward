import unittest
from openreward.rewards.countdown_reward import (
    CountdownReward,
    CountdownRewardConfig,
    CountdownGroundTruth,
)
from openreward.types.chat import Chat, ChatCompletion


class TestCountdownReward(unittest.TestCase):

    def setUp(self):
        self.config = CountdownRewardConfig()
        self.reward = CountdownReward(self.config)

    def test_valid_expression_correct_solution(self):
        chat = Chat(
            messages=[
                ChatCompletion(role="user", content="Solve the Countdown problem."),
                ChatCompletion(role="assistant", content="<answer>2 + 3</answer>"),
            ],
            metadata={"ground_truth": {"target": 5, "starting_numbers": [2, 3]}},
        )
        self.assertEqual(self.reward(chat), 1.0)

    def test_valid_expression_incorrect_solution(self):
        chat = Chat(
            messages=[
                ChatCompletion(role="user", content="Solve the Countdown problem."),
                ChatCompletion(role="assistant", content="<answer>2 + 2</answer>"),
            ],
            metadata={"ground_truth": {"target": 5, "starting_numbers": [2, 3]}},
        )
        self.assertEqual(self.reward(chat), 0.0)

    def test_invalid_expression(self):
        chat = Chat(
            messages=[
                ChatCompletion(role="user", content="Solve the Countdown problem."),
                ChatCompletion(role="assistant", content="<answer>2 + x</answer>"),
            ],
            metadata={"ground_truth": {"target": 5, "starting_numbers": [2, 3]}},
        )
        self.assertEqual(self.reward(chat), 0.0)

    def test_no_expression(self):
        chat = Chat(
            messages=[
                ChatCompletion(role="user", content="Solve the Countdown problem."),
                ChatCompletion(role="assistant", content="I don't know the answer."),
            ],
            metadata={"ground_truth": {"target": 5, "starting_numbers": [2, 3]}},
        )
        self.assertEqual(self.reward(chat), 0)

    def test_valid_format_reward(self):
        self.config.valid_format_reward = 0.5
        self.reward = CountdownReward(self.config)
        chat = Chat(
            messages=[
                ChatCompletion(role="user", content="Solve the Countdown problem."),
                ChatCompletion(role="assistant", content="<answer>2 + 2</answer>"),
            ],
            metadata={"ground_truth": {"target": 5, "starting_numbers": [2, 3]}},
        )
        self.assertEqual(self.reward(chat), 0.5)


if __name__ == "__main__":
    unittest.main()
