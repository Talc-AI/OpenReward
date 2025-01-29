from typing import List, Optional, Literal, Any
from pydantic import BaseModel


class ChatCompletion(BaseModel):
    content: Optional[str] = None
    role: Literal["assistant"] | Literal["user"]


class Chat(BaseModel):
    messages: List[ChatCompletion]


def parse_chat(chat: Any) -> Chat:
    return Chat(messages=[ChatCompletion(**message) for message in chat["messages"]])
