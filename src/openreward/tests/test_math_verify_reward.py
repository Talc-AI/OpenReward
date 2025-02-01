import unittest
from openreward.rewards.math_verify_reward import (
    MathVerifyReward,
    MathVerifyRewardConfig,
)
from openreward.types.chat import Chat, ChatCompletion


class TestMathVerifyReward(unittest.TestCase):

    def setUp(self):
        self.default_config = MathVerifyRewardConfig(valid_expression_reward=0.5)
        self.custom_config = MathVerifyRewardConfig(
            extraction_regex=r"<solution>(.*?)</solution>",
            valid_expression_reward=0.5,
        )

    def test_valid_expression_correct_solution(self):
        reward = MathVerifyReward(self.default_config)
        chat = Chat(
            messages=[
                ChatCompletion(role="user", content="Solve the math problem 2 + 3."),
                ChatCompletion(
                    role="assistant",
                    content="<thinking>2 + 3 = 5<thinking><answer>5</answer>",
                ),
            ],
            metadata={"ground_truth": 5},
        )
        self.assertEqual(reward(chat), 1.0)

    def test_valid_expression_correct_solution_newline(self):
        reward = MathVerifyReward(self.default_config)
        chat = Chat(
            messages=[
                ChatCompletion(role="user", content="Solve the math problem 2 + 3."),
                ChatCompletion(
                    role="assistant",
                    content="""<reasoning>
First, calculate the weekly spending on lattes: 1 latte per day for 7 days, so $4/week. 
Lattes 52 weeks = 52*4 = $208, 
She buys an iced coffee for 3 days a week. 
The average is (3+2)/2 = 2.5, 
Thus, she spends 2.5*3 = 7.5 on iced coffee weekly, 
The total weekly spending on coffee is 7.5 + 4.00 = 11.5.
The yearly spending is 11.5*52 = 598.0 dollars.
Her goal is to cut her spending by 25%, 0.25*598.0 = 149.5 dollars, 
Therefore, she will save 598.0 - 149.5 = 448
</reasoning>

<answer>
448
</answer>""",
                ),
            ],
            metadata={"ground_truth": 448},
        )
        self.assertEqual(reward(chat), 1.0)

    def test_valid_expression_incorrect_solution(self):
        reward = MathVerifyReward(self.default_config)
        chat = Chat(
            messages=[
                ChatCompletion(role="user", content="Solve the math problem 2 + 3."),
                ChatCompletion(
                    role="assistant",
                    content="<thinking>2 + 3 = 6<thinking><answer>6</answer>",
                ),
            ],
            metadata={"ground_truth": 5},
        )
        self.assertEqual(reward(chat), 0.5)

    def test_invalid_expression(self):
        reward = MathVerifyReward(self.default_config)
        chat = Chat(
            messages=[
                ChatCompletion(role="user", content="Solve the math problem 2 + 3."),
                ChatCompletion(
                    role="assistant",
                    content="<thinking>2 + 3 = <thinking><answer>five</answer>",
                ),
            ],
            metadata={"ground_truth": 5},
        )
        self.assertEqual(reward(chat), 0)

    def test_valid_expression_correct_solution_custom_regex(self):
        reward = MathVerifyReward(self.custom_config)
        chat = Chat(
            messages=[
                ChatCompletion(role="user", content="Solve the math problem 2 + 3."),
                ChatCompletion(
                    role="assistant",
                    content="<thinking>2 + 3 = 5<thinking><solution>5</solution>",
                ),
            ],
            metadata={"ground_truth": 5},
        )
        self.assertEqual(reward(chat), 1.0)

    def test_valid_expression_incorrect_solution_custom_regex(self):
        reward = MathVerifyReward(self.custom_config)
        chat = Chat(
            messages=[
                ChatCompletion(role="user", content="Solve the math problem 2 + 3."),
                ChatCompletion(
                    role="assistant",
                    content="<thinking>2 + 3 = 6<thinking><solution>6</solution>",
                ),
            ],
            metadata={"ground_truth": 5},
        )
        self.assertEqual(reward(chat), 0.5)

    def test_no_valid_expression(self):
        reward = MathVerifyReward(self.default_config)
        chat = Chat(
            messages=[
                ChatCompletion(role="user", content="Solve the math problem 2 + 3."),
                ChatCompletion(
                    role="assistant",
                    content="<thinking>2 + 3 = <thinking><answer></answer>",
                ),
            ],
            metadata={"ground_truth": 5},
        )
        self.assertEqual(reward(chat), 0.0)


if __name__ == "__main__":
    unittest.main()
