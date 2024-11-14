from typing import Optional
from logging_utils import logger
from telegram.ext import ContextTypes


class DefaultStrategy:

    async def init_strategy(self, context: ContextTypes.DEFAULT_TYPE) -> Optional[bool]:
        try:
            logger.info('Running default strategy')
            return True
        except Exception as e:
            logger.error(f"Error executing default strategy: {str(e)}")
            return False
