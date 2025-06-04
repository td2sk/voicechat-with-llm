from logging import getLogger

import pyaudio

logger = getLogger(__name__)


def get_input_devices():
    audio = pyaudio.PyAudio()
    host_api = audio.get_default_host_api_info()
    devices = []
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        # 出力デバイスは無視
        if info["maxInputChannels"] == 0:
            continue
        if info["hostApi"] != host_api["index"]:
            continue
        devices.append(info)
    return devices


if __name__ == "__main__":
    devices = get_input_devices()
    for device in devices:
        print("id: %d, name: %s" % (device["index"], device["name"]))
