import os
import requests
from typing import Tuple
from logging_utils import logger


class OpenRouterClient:
    """Client for interacting with OpenRouter.ai API."""
    
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    SYSTEM_PROMPT = (
        "You are a professional cryptocurrency technical analyst with expertise in "
        "multi-timeframe analysis and comprehensive indicator usage. Analyze all provided "
        "technical indicators including trend, momentum, volatility, and volume indicators "
        "for complete market assessment. Respond ONLY with valid JSON format without any "
        "additional text or formatting."
    )
    
    def __init__(self):
        self.api_key = os.getenv("HTB_OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("HTB_OPENROUTER_API_KEY environment variable not set")
    
    def call_api(self, model: str, prompt: str) -> str:
        """Call OpenRouter.ai API for AI analysis and return response."""
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "reasoning": {"max_tokens": 10000},
            "response_format": {"type": "json_object"},
            "usage": {"include": "true"},
            "max_tokens": 17500
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                self.BASE_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]

            return content.strip().removeprefix("```json").removesuffix("```").strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API request failed: {str(e)}", exc_info=True)
            raise ValueError(f"OpenRouter API request failed: {str(e)}")
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse API response: {str(e)}\nResponse: {data}", exc_info=True)
            raise ValueError(f"Invalid API response format: {str(e)}")
