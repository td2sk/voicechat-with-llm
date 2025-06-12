import numpy as np
import requests

from audio.transcribe.base import BaseTranscriber, TranscribeError


class RemoteTranscriber(BaseTranscriber):
    def __init__(self, endpoint: str, beam_size: int, language: str | None):
        super().__init__()
        self._endpoint = endpoint.rstrip("/")
        self._beam_size = beam_size
        self._language = language

    def transcribe(self, audio: np.ndarray) -> str:
        try:
            response = requests.post(
                self._endpoint + "/transcribe",
                files={"audio": audio.tobytes()},
                params={"beam_size": self._beam_size, "language": self._language},
            )
            response.raise_for_status()
            return response.json()["text"]
        except Exception as e:
            raise TranscribeError from e
