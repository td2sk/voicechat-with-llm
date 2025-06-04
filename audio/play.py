import io
import time
import wave
from logging import getLogger

import pyaudio

logger = getLogger(__name__)


def play_wav(voice: bytes, delay: float = 0.2):
    with io.BytesIO(voice) as b, wave.open(b) as wav:
        logger.debug(
            f"wave info: {wav.getsampwidth()}, {wav.getnchannels()}, {wav.getframerate()}"
        )
        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(wav.getsampwidth()),
            channels=wav.getnchannels(),
            rate=wav.getframerate(),
            output=True,
        )
        try:
            chunk = 1024
            data = wav.readframes(chunk)
            while data:
                stream.write(data)
                data = wav.readframes(chunk)

            # 音声が尻切れにならないよう若干の delay を入れる
            time.sleep(delay)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
