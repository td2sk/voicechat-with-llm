import json
import threading
from logging import getLogger
from queue import Queue

import numpy as np
import pyaudio

import utils.log_duration as log_duration
from audio.play import play_wav
from audio.stream import AudioStream
from audio.transcribe import Transcriber
from audio.vad import Vad
from llm import chat
from voice.voicevox import VOICEVOX

logger = getLogger(__name__)


def run_chat(client: chat.Client, message_queue: Queue[str], talk_queue: Queue):
    while True:
        text = message_queue.get()
        with log_duration.info("llm chat"):
            response = client.chat(text)
        logger.info("LLM: %s", response.message.content)
        content = json.loads(response.message.content)
        talk_queue.put(content)


# TODO 四国めたん前提となっている
TONE_MAP = {
    "普通": 2,
    "あまあま": 0,
    "ツンツン": 6,
    "セクシー": 4,
    "ささやき": 36,
    "ヒソヒソ": 37,
}


def run_voice(voicevox: VOICEVOX, talk_queue: Queue, mute_queue: Queue):
    while True:
        content = talk_queue.get()
        tone = content["tone"]
        speaker = TONE_MAP[tone]
        with log_duration.info("audio_query"):
            query = voicevox.audio_query(speaker, content["content"])
        with log_duration.info("synthesis"):
            voice_wav = voicevox.synthesis(speaker, query)
        mute_queue.put("mute")
        play_wav(voice_wav)
        mute_queue.put("unmute")


def vad(vad_mode: int, device_id: int, audio_queue: Queue[np.typing.NDArray]):
    def vad_callback(audio: np.typing.NDArray):
        logger.info("voice detected")
        audio_queue.put(audio)

    vad = Vad(vad_callback, mode=vad_mode)

    def stream_callback(in_data: bytes, frames: int, time_info: dict, status: int):
        vad.process(in_data)
        return (in_data, pyaudio.paContinue)

    return AudioStream(device_id, stream_callback)


def transcribe(
    transcriber: Transcriber,
    audio_queue: Queue[np.typing.NDArray],
    message_queue: Queue[str],
):
    while True:
        audio = audio_queue.get()
        with log_duration.info("transcribe time"):
            segments = transcriber.transcribe(audio)
            segments = list(segments)
        for segment in segments:
            logger.info("transcribed: %s", segment.text)
            message_queue.put(segment.text)


def main():
    import argparse
    import os
    from logging import INFO, WARN, basicConfig, getLogger

    basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        level=INFO,
    )
    getLogger("faster_whisper").setLevel(WARN)

    parser = argparse.ArgumentParser()
    parser.add_argument("--system-prompt-path", type=str)
    parser.add_argument("--schema-path", type=str)
    parser.add_argument("--ollama-host", type=str, default="127.0.0.1:11434")
    parser.add_argument("--ollama-model", type=str)
    parser.add_argument("--whisper-model", type=str, default="turbo")
    parser.add_argument("--whisper-device", type=str)
    parser.add_argument("--whisper-type", type=str, default="int8")
    parser.add_argument("--whisper-beam-size", type=int, default=1)
    parser.add_argument("--whisper-language", type=str)
    parser.add_argument("--vad-mode", type=int, default=3)
    parser.add_argument("--audio-device-id", type=int, default=0)
    args = parser.parse_args()

    if args.ollama_host:
        ollama_host = args.ollama_host
    else:
        host_from_env = os.getenv("OLLAMA_HOST")
        if host_from_env is None:
            logger.error("--ollama-host or OLLAMA_HOST are not set")

    audio_queue: Queue[np.typing.NDArray] = Queue()
    message_queue: Queue[str] = Queue()
    voice_queue: Queue = Queue()
    mute_queue: Queue = Queue()

    with open(args.schema_path, "rb") as f:
        schema = json.load(f)

    with open(args.system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    client = chat.Client(
        model=args.ollama_model, host=ollama_host, system=system_prompt, schema=schema
    )

    voicevox = VOICEVOX()

    transcriber = Transcriber(
        args.whisper_model,
        args.whisper_device,
        args.whisper_type,
        args.whisper_beam_size,
        args.whisper_language,
    )

    threading.Thread(
        target=run_chat,
        daemon=True,
        args=(client, message_queue, voice_queue),
    ).start()

    threading.Thread(
        target=run_voice,
        daemon=True,
        args=(voicevox, voice_queue, mute_queue),
    ).start()

    threading.Thread(
        target=transcribe,
        daemon=True,
        args=(transcriber, audio_queue, message_queue),
    ).start()

    stream = vad(args.vad_mode, args.audio_device_id, audio_queue)

    def audio_thread():
        stream.start()
        logger.info("start listening...")
        while True:
            msg = mute_queue.get(timeout=1000)
            if msg == "mute":
                stream.stop()
            elif msg == "unmute":
                stream.start()

    threading.Thread(target=audio_thread, daemon=True)

    try:
        while True:
            key = input("話しかけてください。キー入力で終了します...\n")
            if not key:
                break
    except KeyboardInterrupt:
        logger.info("interrupt")
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main()
