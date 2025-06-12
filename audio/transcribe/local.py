from logging import getLogger

import numpy as np
from faster_whisper import WhisperModel

import utils.log_duration as log_duration
from audio.transcribe.base import BaseTranscriber

logger = getLogger(__name__)


class LocalTranscriber(BaseTranscriber):
    def __init__(
        self,
        model: str = "turbo",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
        language: str | None = "ja",
    ):
        super().__init__()
        self.beam_size = beam_size
        self.language = language
        logger.info("start loadind whisper model: %s", model)
        with log_duration.info("finish loading whisper model"):
            self.model = WhisperModel(model, device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio,
            self.language,
            beam_size=self.beam_size,
            without_timestamps=True,
        )
        return "".join(segment.text for segment in segments)
