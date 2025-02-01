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
        prompts = [prompt[-1]["content"] for prompt in prompts]

        chats = [
            Chat(
                messages=[
                    ChatCompletion(role="user", content=prompt),
                    ChatCompletion(role="assistant", content=response),
                ],
                metadata={"ground_truth": gt},
            )
            for prompt, response, gt in zip(prompts, responses, answer)
        ]

        rewards = [reward(chat) for chat in chats]

        rewards = [reward if reward is not None else 0.0 for reward in rewards]

        for i in range(len(rewards)):
            print(
                "-" * 20,
                f"Prompt:\n{prompts[i]}",
                f"\nAnswer:\n{answer[i]}",
                f"\nResponse:\n{responses[i]}",
                f"\nReward:\n{rewards[i]}",
            )

        return rewards

    return trl_reward_func
