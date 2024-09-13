from pydantic import BaseModel


class Chat(BaseModel):
    human_message: str
