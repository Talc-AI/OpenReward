# Orignal code written by Will Brown
# Source: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb

import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from openreward.rewards.math_verify_reward import (
    MathVerifyReward,
    MathVerifyRewardConfig,
)
from openreward.rewards.thinking_format_reward import (
    ThinkingFormatReward,
    ThinkingFormatRewardConfig,
)
from openreward.rewards.composite_reward import CompositeReward, CompositeRewardConfig
from openreward.integrations.trl import to_trl

# Load and prep dataset

SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]  # type: ignore
    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "What is the largest single-digit prime number?",
                },
                {
                    "role": "assistant",
                    "content": XML_COT_FORMAT.format(
                        reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
                        answer="7",
                    ),
                },
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )  # type: ignore
    return data  # type: ignore


dataset = get_gsm8k_questions()

eval_dataset = get_gsm8k_questions("test")

reward_funcs = [
    # to_trl(MathVerifyReward(MathVerifyRewardConfig(valid_expression_reward=0.5))),
    # to_trl(ThinkingFormatReward(ThinkingFormatRewardConfig())),
    to_trl(
        CompositeReward(
            CompositeRewardConfig(
                mode="sum",
                rewards=[
                    MathVerifyRewardConfig(valid_expression_reward=0.5),
                    ThinkingFormatRewardConfig(),
                ],
            )
        )
    ),
]


model_name = "meta-llama/Llama-3.2-1B-Instruct"

output_dir = "/models/Llama-1B-GRPO"
run_name = "Llama-1B-GRPO-gsm8k"

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_generations=4,
    max_prompt_length=256,
    max_completion_length=256,
    num_train_epochs=4,
    save_strategy="epoch",
    max_grad_norm=0.1,
    report_to="wandb",
    log_on_each_node=False,
    # use_vllm=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=None,
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# use peft at your own risk; not working for me with multi-GPU training
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_funcs,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
