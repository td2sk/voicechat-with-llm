from logging import getLogger
from typing import Callable

import pyaudio

logger = getLogger(__name__)


class AudioStream:
    def __init__(
        self,
        device_index: int,
        callback: Callable[[tuple[bytes, int, dict, int]], tuple[bytes, int]],
        rate: int = 16000,
        chunk: int = 480,
    ):
        self.stream = self._create_stream(device_index, callback, rate, chunk)

    def start(self):
        self.stream.start_stream()

    def stop(self):
        self.stream.stop_stream()

    def close(self):
        self.stream.close()

    def _create_stream(
        self, device_index: int, callback, rate: int, chunk: int, channels: int = 1
    ):
        FORMAT = pyaudio.paInt16

        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=FORMAT,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk,
            stream_callback=callback,
        )

        return stream


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        level=logging.DEBUG,
    )

    def callback(in_data: bytes, frames: int, time: dict, status: int):
        return (in_data, pyaudio.paContinue)

    stream = AudioStream(0, callback)
    stream.start()
    import time

    time.sleep(100)
