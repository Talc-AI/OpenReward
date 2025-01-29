import unittest
from openreward.thinking_format_reward import (
    ThinkingFormatReward,
    ThinkingFormatRewardConfig,
)
from openreward.types.chat import Chat, Message


class TestThinkingFormatReward(unittest.TestCase):

    def setUp(self):
        self.default_config = ThinkingFormatRewardConfig()
        self.custom_config = ThinkingFormatRewardConfig(
            override_regex=r"<thought>(.*?)</thought><response>(.*?)</response>",
        )

    def test_default_format_match(self):
        reward = ThinkingFormatReward(self.default_config)
        chat = Chat(
            messages=[
                Message(role="user", content="What is the capital of France?"),
                Message(
                    role="assistant",
                    content="<thinking>Let me think...</thinking><answer>Paris</answer>",
                ),
            ]
        )
        self.assertEqual(reward(chat), 1.0)

    def test_default_format_no_match(self):
        reward = ThinkingFormatReward(self.default_config)
        chat = Chat(
            messages=[
                Message(role="user", content="What is the capital of France?"),
                Message(role="assistant", content="The capital of France is Paris."),
            ]
        )
        self.assertEqual(reward(chat), 0.0)

    def test_custom_format_match(self):
        reward = ThinkingFormatReward(self.custom_config)
        chat = Chat(
            messages=[
                Message(role="user", content="What is the capital of France?"),
                Message(
                    role="assistant",
                    content="<thought>Let me think...</thought><response>Paris</response>",
                ),
            ]
        )
        self.assertEqual(reward(chat), 1.0)

    def test_custom_format_no_match(self):
        reward = ThinkingFormatReward(self.custom_config)
        chat = Chat(
            messages=[
                Message(role="user", content="What is the capital of France?"),
                Message(
                    role="assistant",
                    content="<thinking>Let me think...</thinking><answer>Paris</answer>",
                ),
            ]
        )
        self.assertEqual(reward(chat), 0.0)

    def test_insufficient_messages(self):
        reward = ThinkingFormatReward(self.default_config)
        chat = Chat(
            messages=[Message(role="user", content="What is the capital of France?")]
        )
        self.assertIsNone(reward(chat))

    def test_last_message_not_assistant(self):
        reward = ThinkingFormatReward(self.default_config)
        chat = Chat(
            messages=[
                Message(role="user", content="What is the capital of France?"),
                Message(
                    role="user",
                    content="Actually, I meant to ask about the capital of Germany.",
                ),
            ]
        )
        self.assertIsNone(reward(chat))


if __name__ == "__main__":
    unittest.main()
