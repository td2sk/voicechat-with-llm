from logging import getLogger

from faster_whisper import WhisperModel

import utils.log_duration as log_duration

logger = getLogger(__name__)


class Transcriber:
    def __init__(
        self,
        model: str = "turbo",
        device: str = "cpu",
        compute_type="int8",
        beam_size: int = 1,
    ):
        self.beam_size = beam_size
        logger.info("start loadind whisper model: %s", model)
        with log_duration.info("finish loading whisper model"):
            self.model = WhisperModel(model, device, compute_type=compute_type)

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(
            audio,
            "ja",
            beam_size=self.beam_size,
            without_timestamps=True,
        )
        return segments


if __name__ == "__main__":
    from logging import INFO, basicConfig

    basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        level=INFO,
    )
    transcriber = Transcriber(model="turbo")
