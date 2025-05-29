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
    
    def call_api(self, model: str, prompt: str) -> Tuple[str, float]:
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
                    "content": "You are a professional cryptocurrency technical analyst specializing in intraday trading. Provide clear, actionable analysis based on technical indicators and market data. . Respond ONLY with valid JSON format. Do not include any text outside the JSON structure."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "reasoning": {
                "max_tokens": 5000
            },
            "response_format": {
                "type": "json_object"
            },
            "usage": {
                "include": "true"
            },
            "max_tokens": 10000
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
            
            # Extract actual cost from OpenRouter response
            usage = data.get("usage", {})
            total_cost = usage.get("cost", 0.0)
            
            return data["choices"][0]["message"]["content"], total_cost
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"OpenRouter API request failed: {str(e)}")