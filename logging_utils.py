import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S'
)

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

apscheduler = logging.getLogger("apscheduler.scheduler")
apscheduler.setLevel(logging.WARNING)

apscheduler_executors = logging.getLogger("apscheduler.executors.default")
apscheduler_executors.setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
