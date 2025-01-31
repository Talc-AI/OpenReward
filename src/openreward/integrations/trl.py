from openreward.rewards.reward import Reward
from openreward.types.chat import Chat, ChatCompletion


# An example TRL reward function
# def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
#     responses = [completion[0]["content"] for completion in completions]
#     q = prompts[0][-1]["content"]
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     print(
#         "-" * 20,
#         f"Question:\n{q}",
#         f"\nAnswer:\n{answer[0]}",
#         f"\nResponse:\n{responses[0]}",
#         f"\nExtracted:\n{extracted_responses[0]}",
#     )
#     return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def to_trl(
    reward: Reward,
):
    """Wraps the reward function to be compatible with the TRL framework."""

    def trl_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        responses = [completion[0]["content"] for completion in completions]
        prompts = [prompts[-1] for prompt in prompts]

        chats = [
            Chat(
                messages=[
                    ChatCompletion(role="user", content=prompt),
                    ChatCompletion(role="assistant", content=response),
                ],
                metadata={"ground_truth": answer},
            )
            for prompt, response in zip(prompts, responses)
        ]

        return [reward(chat) for chat in chats]

    return trl_reward_func
