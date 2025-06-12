import json
import threading
from logging import getLogger
from queue import Queue

import numpy as np
import pyaudio

import utils.log_duration as log_duration
from audio.play import play_wav
from audio.stream import AudioStream
from audio.transcribe.base import BaseTranscriber
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
        if response.message.content is None:
            continue
        content = json.loads(response.message.content)
        talk_queue.put(content)


def run_voice_synthesis(
    voicevox: VOICEVOX, styles, talk_queue: Queue, voice_play_queue: Queue
):
    while True:
        content = talk_queue.get()
        tone = content["tone"]

        speaker = next((style["id"] for style in styles if style["name"] == tone), None)
        if speaker is None:
            logger.warning(
                "unknown tone: %s. use %s instead", (tone, styles[0]["name"])
            )
            speaker = styles[0]["id"]
        with log_duration.info("audio_query"):
            query = voicevox.audio_query(speaker, content["content"])
        with log_duration.info("synthesis"):
            voice_wav = voicevox.synthesis(speaker, query)
        voice_play_queue.put(voice_wav)


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
    transcriber: BaseTranscriber,
    audio_queue: Queue[np.typing.NDArray],
    message_queue: Queue[str],
):
    while True:
        audio = audio_queue.get()
        with log_duration.info("transcribe time"):
            text = transcriber.transcribe(audio)
        logger.info("transcribed: %s", text)
        message_queue.put(text)


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
    parser.add_argument("--system-prompt-path", type=str, required=True)
    parser.add_argument("--ollama-host", type=str, default="127.0.0.1:11434")
    parser.add_argument("--ollama-model", type=str, required=True)
    parser.add_argument(
        "--voicevox-endpoint", type=str, default="http://127.0.0.1:50021"
    )
    parser.add_argument("--voicevox-character", type=str, default="四国めたん")
    parser.add_argument("--vad-mode", type=int, default=3)
    parser.add_argument("--audio-device-id", type=int, default=0)
    parser.add_argument("--whisper-mode", choices=["local", "remote"], default="local")
    remote_parser = parser.add_argument_group("whisper-mode-remote")
    remote_parser.add_argument(
        "--whisper-endpoint", type=str, default="http://127.0.0.1:8000"
    )
    local_parser = parser.add_argument_group("whisper-mode-local")
    local_parser.add_argument("--whisper-model", type=str, default="turbo")
    local_parser.add_argument("--whisper-device", type=str)
    local_parser.add_argument("--whisper-type", type=str, default="int8")
    parser.add_argument("--whisper-beam-size", type=int, default=1)
    parser.add_argument("--whisper-language", type=str)
    args = parser.parse_args()

    ollama_host = None
    if args.ollama_host:
        ollama_host = args.ollama_host
    else:
        host_from_env = os.getenv("OLLAMA_HOST")
        if host_from_env is None:
            logger.error("--ollama-host or OLLAMA_HOST are not set")

    audio_queue: Queue[np.typing.NDArray] = Queue()
    message_queue: Queue[str] = Queue()
    talk_quele: Queue = Queue()
    voice_play_queue: Queue = Queue()

    voicevox = VOICEVOX(args.voicevox_endpoint)
    speakers: list = json.loads(voicevox.speakers())

    cv = next((r for r in speakers if r["name"] == args.voicevox_character), None)
    if cv is None:
        logger.error("character not found: %s", args.voicevox_character)
        return
    logger.info("character voice: %s", cv["name"])
    styles = cv["styles"]

    llm_response_schema = {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "tone": {
                "type": "string",
                "enum": list(style["name"] for style in styles),
            },
        },
        "required": ["content", "tone"],
        "additionalProperties": False,
    }

    with open(args.system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
        system_prompt = system_prompt.replace(
            "{%DEFAULT_OUTPUT_FORMAT}",
            """- json 形式で出力してください
- content 属性には、会話の内容を指定してください
- ユーザー側のセリフやナレーションは書かないでください
- tone 属性では、あなたの話のトーン、感情を指定します。返事の内容に沿って以下の中から選んでください
{%VOICEVOX_TONES}
""",
        )
        system_prompt = system_prompt.replace(
            "{%VOICEVOX_CHARACTER}", args.voicevox_character
        )
        system_prompt = system_prompt.replace(
            "{%VOICEVOX_TONES}", "\n".join(f"  - {style["name"]}" for style in styles)
        )
        logger.info("prompt:\n%s", system_prompt)

    client = chat.Client(
        model=args.ollama_model,
        host=ollama_host,
        system=system_prompt,
        schema=llm_response_schema,
    )

    if args.whisper_mode == "remote":
        from audio.transcribe.remote import RemoteTranscriber

        transcriber = RemoteTranscriber(
            args.whisper_endpoint,
            args.whisper_beam_size,
            args.whisper_language,
        )
    else:
        from audio.transcribe.local import LocalTranscriber

        transcriber = LocalTranscriber(
            args.whisper_model,
            args.whisper_device,
            args.whisper_type,
            args.whisper_beam_size,
            args.whisper_language,
        )

    threading.Thread(
        target=run_chat,
        daemon=True,
        args=(client, message_queue, talk_quele),
    ).start()

    threading.Thread(
        target=run_voice_synthesis,
        daemon=True,
        args=(voicevox, styles, talk_quele, voice_play_queue),
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
            try:
                voice = voice_play_queue.get(timeout=1000)
                stream.stop()
                play_wav(voice)
                stream.start()
            except TimeoutError:
                pass

    threading.Thread(
        target=audio_thread,
        daemon=True,
    ).start()

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
