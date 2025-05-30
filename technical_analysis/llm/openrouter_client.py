import os
import requests
from typing import Tuple
from logging_utils import logger


class OpenRouterClient:
    """Client for interacting with OpenRouter.ai API."""
    
    def __init__(self):
        self.api_key = os.getenv("HTB_OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError("HTB_OPENROUTER_API_KEY environment variable not set")
    
    def call_api(self, model: str, prompt: str) -> str:
        """Call OpenRouter.ai API for AI analysis and return response with cost."""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional cryptocurrency technical analyst with expertise in multi-timeframe analysis and comprehensive indicator usage. Analyze all provided technical indicators including trend, momentum, volatility, and volume indicators for complete market assessment. Respond ONLY with valid JSON format without any additional text or formatting."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "reasoning": {
                "max_tokens": 10000
            },
            "response_format": {
                "type": "json_object"
            },
            "usage": {
                "include": "true"
            },
            "max_tokens": 17500
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise ValueError(f"OpenRouter API error {response.status_code}: {response.text}")

            data = response.json()
            print(data)
            # Clean up the content by removing markdown code block wrapper if present
            content = data["choices"][0]["message"]["content"]
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            return content
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"OpenRouter API request failed: {str(e)}")