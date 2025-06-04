import logging
import time

logger = logging.getLogger(__name__)


class log_duration:
    def __init__(self, message_fmt: str, level: int, *args):
        self.message_fmt = message_fmt
        self.args = args
        self.level = level

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start
        if exc_type:
            logger.warning(
                "%s failed after %.3f sec (exception %s)",
                self.message_fmt % self.args,
                duration,
                exc_type.__name__,
            )
        if logger.isEnabledFor(self.level):
            logger.log(
                self.level, "%s: %.3f sec", self.message_fmt % self.args, duration
            )


def debug(message_fmt: str, *args):
    return log_duration(message_fmt, level=logging.DEBUG, *args)


def info(message_fmt: str, *args):
    return log_duration(message_fmt, level=logging.INFO, *args)
