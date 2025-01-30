from typing import List, Optional, Literal, Any
from pydantic import BaseModel


class ChatCompletion(BaseModel):
    content: Optional[str] = None
    role: Literal["assistant"] | Literal["user"] | Literal["system"]


class Metadata(BaseModel, extra="allow"):
    ground_truth: Optional[Any] = None
    dataset: Optional[str] = None


class Chat(BaseModel):
    messages: List[ChatCompletion]
    metadata: Optional[Metadata] = None


def parse_chat(
    chat: Any, prompt: str | None = None, ground_truth: Any | None = None
) -> Chat:
    if isinstance(chat, Chat):
        parsed = chat
        if ground_truth is not None:
            if parsed.metadata is None:
                parsed.metadata = Metadata(ground_truth=ground_truth)
            else:
                parsed.metadata.ground_truth = ground_truth
        return parsed
    if isinstance(chat, str):
        messages = []
        if prompt is not None:
            messages.append(ChatCompletion(content=prompt, role="system"))
        messages.append(ChatCompletion(content=chat, role="assistant"))
        metadata = Metadata(ground_truth=ground_truth)
        return Chat(messages=messages, metadata=metadata)
    return Chat(
        messages=[ChatCompletion(**message) for message in chat["messages"]],
        metadata=Metadata(**chat.get("metadata", {"ground_truth": ground_truth})),
    )
