import os
import litellm
from typing import Tuple
from logging_utils import logger, logging


class LiteLLMClient:
    """Client for interacting with various LLM providers through LiteLLM."""
    
    SYSTEM_PROMPT = (
        "You are a professional cryptocurrency technical analyst with expertise in "
        "multi-timeframe analysis and comprehensive indicator usage. Analyze all provided "
        "technical indicators including trend, momentum, volatility, and volume indicators "
        "for complete market assessment. Respond ONLY with valid JSON format without any "
        "additional text or formatting."
    )
    
    def __init__(self):
        litellm.use_litellm_proxy = True
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    
    async def call_api(self, model: str, prompt: str) -> str:
        """Call LiteLLM API for AI analysis and return response."""
        
        try:
            response = await litellm.acompletion(
                model=model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=17500,
                timeout=60
            )
            
            content = response.choices[0].message.content
            return content.strip().removeprefix("```json").removesuffix("```").strip()
            
        except Exception as e:
            logger.error(f"LiteLLM API request failed: {str(e)}", exc_info=True)
            raise ValueError(f"LiteLLM API request failed: {str(e)}")