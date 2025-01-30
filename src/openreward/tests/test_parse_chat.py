import unittest
from pydantic import ValidationError
from openreward.types.chat import Chat, ChatCompletion, Metadata, parse_chat


class TestParseChat(unittest.TestCase):

    def test_parse_chat_with_chat_instance(self):
        chat_instance = Chat(
            messages=[ChatCompletion(content="Hello", role="user")],
            metadata=Metadata(ground_truth="test_ground_truth"),
        )
        parsed_chat = parse_chat(chat_instance)
        self.assertEqual(parsed_chat, chat_instance)

    def test_parse_chat_with_chat_instance_and_ground_truth(self):
        chat_instance = Chat(
            messages=[ChatCompletion(content="Hello", role="user")],
            metadata=Metadata(dataset="test_dataset"),
        )
        parsed_chat = parse_chat(chat_instance, ground_truth="new_ground_truth")
        self.assertEqual(parsed_chat.metadata.ground_truth, "new_ground_truth")

    def test_parse_chat_with_string(self):
        chat_string = "This is a test chat"
        parsed_chat = parse_chat(chat_string)
        self.assertEqual(parsed_chat.messages[0].content, chat_string)
        self.assertEqual(parsed_chat.messages[0].role, "assistant")

    def test_parse_chat_with_string_and_prompt(self):
        chat_string = "This is a test chat"
        prompt = "This is a prompt"
        parsed_chat = parse_chat(chat_string, prompt=prompt)
        self.assertEqual(parsed_chat.messages[0].content, prompt)
        self.assertEqual(parsed_chat.messages[0].role, "system")
        self.assertEqual(parsed_chat.messages[1].content, chat_string)
        self.assertEqual(parsed_chat.messages[1].role, "assistant")


if __name__ == "__main__":
    unittest.main()
