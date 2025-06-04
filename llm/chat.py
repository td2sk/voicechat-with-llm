from logging import getLogger

import ollama

logger = getLogger(__name__)


class Client:
    def __init__(
        self,
        model: str,
        host: str,
        system: str | None = None,
        schema: str | None = None,
    ):
        self.host = host
        self.model = model
        self.client = ollama.Client(host)
        self.system = system
        self.messages = self._initial_message()
        self.schema = schema

    def _initial_message(self):
        if self.system:
            return [{"role": "system", "content": self.system}]
        else:
            return []

    def reset(self):
        self.messages = self._initial_message()

    def chat(self, text: str):
        self.messages.append({"role": "user", "content": text})
        response = self.client.chat(
            model=self.model, messages=self.messages, think=False, format=self.schema
        )
        self.messages.append(response.message)
        return response
