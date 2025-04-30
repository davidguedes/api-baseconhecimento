from dataclasses import dataclass
from datetime import datetime

@dataclass
class Message:
    content: str
    role: str #'user' ou 'assistant'
    timestamp: datetime = datetime.now()