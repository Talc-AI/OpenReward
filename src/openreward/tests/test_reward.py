import unittest
from openreward.rewards.math_verify_reward import MathVerifyReward
from openreward.rewards.thinking_format_reward import ThinkingFormatReward
from openreward.rewards.composite_reward import CompositeReward
from openreward.utils.load_reward import create_reward


class TestRewardLoading(unittest.TestCase):

    def test_create_math_verify_reward(self):
        config = {
            "type": "MathVerify",
            "extraction_regex": r"<answer>(.*?)</answer>",
            "valid_expression_reward": 0.5,
        }
        reward = create_reward(config)
        self.assertIsInstance(reward, MathVerifyReward)
        self.assertEqual(reward.config.type, "MathVerify")
        self.assertEqual(reward.config.extraction_regex, r"<answer>(.*?)</answer>")
        self.assertEqual(reward.config.valid_expression_reward, 0.5)

    def test_create_thinking_format_reward(self):
        config = {
            "type": "ThinkingFormat",
            "override_regex": r"<thought>(.*?)</thought><response>(.*?)</response>",
        }
        reward = create_reward(config)
        self.assertIsInstance(reward, ThinkingFormatReward)
        self.assertEqual(reward.config.type, "ThinkingFormat")
        self.assertEqual(
            reward.config.override_regex,
            r"<thought>(.*?)</thought><response>(.*?)</response>",
        )

    def test_create_composite_reward(self):
        config = {
            "type": "Composite",
            "mode": "sum",
            "rewards": [
                {
                    "type": "MathVerify",
                    "extraction_regex": r"<answer>(.*?)</answer>",
                    "valid_expression_reward": 0.5,
                },
                {
                    "type": "ThinkingFormat",
                    "override_regex": r"<thought>(.*?)</thought><response>(.*?)</response>",
                },
            ],
        }
        reward = create_reward(config)
        self.assertIsInstance(reward, CompositeReward)
        self.assertEqual(reward.config.type, "Composite")
        self.assertEqual(reward.config.mode, "sum")
        self.assertEqual(len(reward.rewards), 2)
        self.assertIsInstance(reward.rewards[0], MathVerifyReward)
        self.assertIsInstance(reward.rewards[1], ThinkingFormatReward)

    def test_create_reward_invalid_type(self):
        config = {"type": "InvalidType"}
        with self.assertRaises(ValueError):
            create_reward(config)


if __name__ == "__main__":
    unittest.main()
