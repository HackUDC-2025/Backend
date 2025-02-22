from loguru import logger
import sys

logger.remove()

logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO",
    colorize=True,
    enqueue=True,
)

logger.add(
    "./app/logs/app.log",
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
    encoding="utf-8",
    enqueue=True,
)

logger.info("🚀 Logger initialized inside Docker!")
