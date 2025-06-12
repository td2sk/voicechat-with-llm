import numpy as np
import requests

from audio.transcribe.base import BaseTranscriber, TranscribeError


class RemoteTranscriber(BaseTranscriber):
    def __init__(self, endpoint: str):
        super().__init__()
        self._endpoint = endpoint.rstrip("/")

    def transcribe(self, audio: np.ndarray) -> str:
        try:
            response = requests.post(
                self._endpoint + "/transcribe", files={"audio": audio.tobytes()}
            )
            response.raise_for_status()
            return response.json()["text"]
        except Exception as e:
            raise TranscribeError from e
