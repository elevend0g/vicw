"""External LLM inference using OpenAI-compatible APIs"""

import logging
import time
import asyncio
from typing import List, Dict
import httpx

from config import (
    LLM_TIMEOUT,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE
)

logger = logging.getLogger(__name__)
metrics_logger = logging.getLogger('vicw.metrics')


class ExternalLLMInference:
    """
    Asynchronous, non-blocking inference using an external OpenAI-compatible API
    (e.g., OpenRouter, OpenAI, local server).
    """
    
    def __init__(self, api_url: str, api_key: str, model_name: str):
        if not api_key:
            raise ValueError("API key must be provided for ExternalLLMInference")
        
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.client: httpx.AsyncClient = None
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"External LLM configured: model={self.model_name}, url={self.api_url}")
    
    async def init(self):
        """Initialize HTTP client"""
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(LLM_TIMEOUT, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        logger.info("External LLM client initialized")
    
    async def generate(self, context: List[Dict[str, str]], max_tokens: int = None, temperature: float = None) -> str:
        """
        Asynchronous generation via HTTP POST request.
        
        Args:
            context: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate (default from config)
            temperature: Sampling temperature (default from config)
        
        Returns:
            Generated text response
        """
        if not self.client:
            raise RuntimeError("LLM client not initialized. Call init() first.")
        
        gen_start_time = time.time()
        
        payload = {
            "model": self.model_name,
            "messages": context,
            "max_tokens": max_tokens or LLM_MAX_TOKENS,
            "temperature": temperature or LLM_TEMPERATURE,
            "stream": False
        }
        
        try:
            response = await self.client.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            
            response_json = response.json()
            
            # Extract generated text (OpenAI-compatible format)
            if 'choices' in response_json and len(response_json['choices']) > 0:
                generated_text = response_json['choices'][0]['message']['content']
            else:
                raise ValueError(f"Unexpected response format: {response_json}")
            
            gen_time = (time.time() - gen_start_time) * 1000
            
            logger.info(f"Generated response in {gen_time:.2f}ms ({len(generated_text)} chars)")
            metrics_logger.info(
                f"LLM_GENERATION | "
                f"time_ms={gen_time:.2f} | "
                f"response_length={len(generated_text)} | "
                f"model={self.model_name}"
            )
            
            return generated_text
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during LLM generation: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.TimeoutException as e:
            logger.error(f"Timeout during LLM generation after {LLM_TIMEOUT}s")
            raise
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            raise
    
    async def generate_with_retry(
        self,
        context: List[Dict[str, str]],
        max_retries: int = 2,
        retry_delay: float = 1.0
    ) -> str:
        """
        Generate with automatic retry on failure.
        
        Args:
            context: List of message dicts
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        
        Returns:
            Generated text response
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return await self.generate(context)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(f"Generation failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Generation failed after {max_retries + 1} attempts")
        
        raise last_error
    
    async def shutdown(self):
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()
            logger.info("External LLM client closed")
    
    def get_model_info(self) -> Dict:
        """Get information about the configured model"""
        return {
            "model_name": self.model_name,
            "api_url": self.api_url,
            "timeout": LLM_TIMEOUT,
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE
        }
