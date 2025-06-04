from logging import getLogger
from typing import Callable

import numpy as np
import webrtcvad

logger = getLogger(__name__)


class Vad:
    def __init__(
        self,
        callback: Callable[[np.typing.NDArray], None],
        mode: int = 3,
        rate: int = 16000,
    ):
        """無音部を判定する

        Args:
            callback: 発話部分の音声データが投入されるキュー
            mode(int): VAD の積極性モード。0~3で指定。大きいほど無音判定しやすくなる
        """
        self.vad = webrtcvad.Vad(mode)
        self.rate = rate
        self._callback = callback
        self._silent_frames = 0
        self._buffer = []
        self._threshold_silent_frames = 20  # TODO
        self._threshold_short_frames = 20

    def process(self, in_data: bytes):
        """音声データを処理し、発話区間を検出してコールバック関数に渡す

        Args:
            in_data (bytes): 入力音声データ
        """
        is_speech = self._is_speech(in_data)
        # 発話中と判定された場合、バッファに音声データを蓄積する
        if is_speech:
            self._silent_frames = 0
            # TODO np.int16 と仮定している
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            self._buffer.append(audio_data)
            return

        # 無音が閾値以下のフレーム数であれば、発話継続中とみなして何もしない
        self._silent_frames += 1
        if self._silent_frames <= self._threshold_silent_frames:
            return

        # バッファ内の音声が短ければ、ノイズとみなして破棄する
        if len(self._buffer) <= self._threshold_short_frames:
            self._buffer.clear()
            return

        # バッファに蓄積された発話内容をコールバック関数に渡す
        data = np.concatenate(self._buffer)
        self._buffer.clear()
        self._callback(data)

    def _is_speech(self, in_data) -> bool:
        """与えられた音声データに音声が含まれているかどうかを判定します

        引数:
            in_data (bytes): 判定する生の音声データ

        戻り値:
            bool: 入力音声データに音声が検出された場合はTrue、そうでない場合はFalse
        """
        return self.vad.is_speech(in_data, self.rate)


if __name__ == "__main__":
    import logging
    import time

    import pyaudio
    from stream import AudioStream

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        level=logging.INFO,
    )

    def vad_callback(data: np.ndarray):
        logger.info("data size: %s", data.size)

    vad = Vad(vad_callback)

    def callback(in_data: bytes, frames: int, time_info: dict, status: int):
        logger.debug("frames: %d", frames)
        vad.process(in_data)
        return (in_data, pyaudio.paContinue)

    stream = AudioStream(0, callback, chunk=480)

    sec = 30
    logging.info("running %d seconds", sec)
    time.sleep(sec)
