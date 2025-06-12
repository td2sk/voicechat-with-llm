from abc import ABC, abstractmethod

import numpy as np


class TranscribeError(Exception):
    pass


class BaseTranscriber(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> str: ...
