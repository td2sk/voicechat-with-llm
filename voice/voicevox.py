from logging import getLogger

import requests

logger = getLogger(__name__)


class VOICEVOX:
    def __init__(self, endpoint: str = "http://127.0.0.1:50021"):
        self.endpoint = endpoint

    def audio_query(self, speaker: int, text: str) -> bytes:
        logger.debug(f"call /audio_query, speaker={speaker}, text={text}")
        response = requests.post(
            self.endpoint + "/audio_query",
            params={
                "speaker": speaker,
                "text": text,
            },
        )

        if response.status_code != 200:
            logger.error(f"failed to create query: {response.content}")
            raise Exception("failed to create query", response.content)

        return response.content

    def synthesis(self, speaker: int, query: bytes):
        logger.debug(f"call /synthesis, speaker={speaker}")
        response = requests.post(
            self.endpoint + "/synthesis", params={"speaker": speaker}, data=query
        )

        if response.status_code != 200:
            logger.error(f"failed to create query: {response.content}")
            raise Exception("failed to create query", response.content)

        return response.content


if __name__ == "__main__":
    import logging

    import audio.play as play
    from utils import log_duration

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        level=logging.INFO,
    )

    speaker_id_metan_normal = 2
    vv = VOICEVOX()
    with log_duration.info("query"):
        query = vv.audio_query(speaker_id_metan_normal, "こんにちは。めたんです")
    with log_duration.info("synthesis"):
        voice = vv.synthesis(speaker_id_metan_normal, query)
    with log_duration.info("play"):
        play.play_wav(voice)
