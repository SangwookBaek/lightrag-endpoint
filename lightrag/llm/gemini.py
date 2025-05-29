import os
import asyncio
import logging
import numpy as np
from typing import Any, Union, AsyncIterator
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from sentence_transformers import SentenceTransformer
from lightrag.utils import EmbeddingFunc, wrap_embedding_func_with_attrs
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

# Custom exception for retry mechanism
class InvalidResponseError(Exception):
    """Custom exception class for triggering retry mechanism"""
    pass

class GeminiRateLimitError(Exception):
    """Custom exception for Gemini rate limit errors"""
    pass

class GeminiConnectionError(Exception):
    """Custom exception for Gemini connection errors"""
    pass

def create_gemini_client(api_key: str | None = None) -> genai.Client:
    """Create a Gemini client with the given configuration.
    
    Args:
        api_key: Gemini API key. If None, uses the GEMINI_API_KEY environment variable.
        
    Returns:
        A Gemini client instance.
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
    
    return genai.Client(api_key=api_key)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(
        retry_if_exception_type(GeminiRateLimitError)
        | retry_if_exception_type(GeminiConnectionError)
        | retry_if_exception_type(InvalidResponseError)
    ),
)
async def gemini_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    api_key: str | None = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> str:
    """Complete a prompt using Gemini's API with caching support.
    
    Args:
        model: The Gemini model to use (e.g., "gemini-1.5-flash", "gemini-1.5-pro").
        prompt: The prompt to complete.
        system_prompt: Optional system prompt to include.
        history_messages: Optional list of previous messages in the conversation.
        api_key: Optional Gemini API key.
        keyword_extraction: Whether this is for keyword extraction.
        **kwargs: Additional keyword arguments for Gemini API.
        
    Returns:
        The completed text.
        
    Raises:
        InvalidResponseError: If the response from Gemini is invalid or empty.
        GeminiConnectionError: If there is a connection error with the Gemini API.
        GeminiRateLimitError: If the Gemini API rate limit is exceeded.
    """
    if history_messages is None:
        history_messages = []
    
    # Create the Gemini client
    try:
        client = create_gemini_client(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to create Gemini client: {e}")
        raise GeminiConnectionError(f"Failed to create Gemini client: {e}")
    
    # Prepare the content for Gemini
    combined_prompt = ""
    
    # Add system prompt if provided
    if system_prompt:
        combined_prompt += f"System: {system_prompt}\n\n"
    
    # Add history messages
    for msg in history_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        combined_prompt += f"{role}: {content}\n"
    
    # Add the current prompt
    combined_prompt += f"User: {prompt}\n\nAssistant:"
    
    # Configure generation settings
    config = types.GenerateContentConfig(
        max_output_tokens=kwargs.get("max_tokens", 8192),
        temperature=kwargs.get("temperature", 0.1),
        top_p=kwargs.get("top_p", 0.9),
        top_k=kwargs.get("top_k", 40),
    )
    
    logger.debug("===== Entering func of Gemini LLM =====")
    logger.debug(f"Model: {model}")
    logger.debug(f"Config: {config}")
    logger.debug(f"Num of history messages: {len(history_messages)}")
    logger.debug("===== Sending Query to Gemini =====")
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=[combined_prompt],
            config=config,
        )
        
        if not response or not response.text:
            logger.error("Invalid response from Gemini API")
            raise InvalidResponseError("Invalid response from Gemini API")
        
        content = response.text.strip()
        
        if not content:
            logger.error("Received empty content from Gemini API")
            raise InvalidResponseError("Received empty content from Gemini API")
        
        logger.debug(f"Response content len: {len(content)}")
        return content
        
    except ClientError as e:
        error_message = str(e).lower()
        if "rate limit" in error_message or "quota" in error_message:
            logger.error(f"Gemini API Rate Limit Error: {e}")
            raise GeminiRateLimitError(f"Gemini API Rate Limit Error: {e}")
        elif "connection" in error_message or "network" in error_message:
            logger.error(f"Gemini API Connection Error: {e}")
            raise GeminiConnectionError(f"Gemini API Connection Error: {e}")
        else:
            logger.error(f"Gemini API Error: {e}")
            raise InvalidResponseError(f"Gemini API Error: {e}")
    except Exception as e:
        logger.error(f"Gemini API Call Failed, Model: {model}, Got: {e}")
        raise InvalidResponseError(f"Gemini API Call Failed: {e}")

async def gemini_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> str:
    """Complete a prompt using the default Gemini model."""
    if history_messages is None:
        history_messages = []
    
    # Get model name from kwargs or use default
    model_name = kwargs.get("model", "gemini-2.0-flash")
    
    return await gemini_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        **kwargs,
    )

async def gemini_2_0_flash_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> str:
    """Complete a prompt using Gemini 1.5 Flash."""
    if history_messages is None:
        history_messages = []
    
    return await gemini_complete_if_cache(
        "gemini-2.0-flash",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        **kwargs,
    )
@wrap_embedding_func_with_attrs(embedding_dim=384, max_token_size=8192)
async def sentence_transformer_embed(
    texts: list[str],
    model: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Generate embeddings using SentenceTransformer.
    
    Args:
        texts: List of texts to embed.
        model: The SentenceTransformer model to use.
        
    Returns:
        A numpy array of embeddings, one per input text.
    """
    # Initialize the model
    st_model = SentenceTransformer(model)
    
    # Generate embeddings
    embeddings = st_model.encode(texts, convert_to_numpy=True)
    
    return embeddings

@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(
        retry_if_exception_type(GeminiRateLimitError)
        | retry_if_exception_type(GeminiConnectionError)
        | retry_if_exception_type(InvalidResponseError)
    ),
)
async def gemini_embed(
    texts: list[str],
    model: str = "gemini-embedding-exp-03-07",
    api_key: str = None,
) -> np.ndarray:
    """Generate embeddings using Gemini's embedding API.
    
    Args:
        texts: List of texts to embed.
        model: The embedding model to use.
        api_key: Optional Gemini API key.
        
    Returns:
        A numpy array of embeddings, one per input text.
        
    Raises:
        InvalidResponseError: If the response from Gemini is invalid.
        GeminiConnectionError: If there is a connection error.
        GeminiRateLimitError: If the API rate limit is exceeded.
    """
    try:
        client = create_gemini_client(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to create Gemini client for embedding: {e}")
        raise GeminiConnectionError(f"Failed to create Gemini client: {e}")
    
    embeddings_list = []
    
    try:
        for text in texts:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding, using default empty embedding")
                # Create a zero embedding with the expected dimension
                embeddings_list.append(np.zeros(768))
                continue
            
            result = client.models.embed_content(
                model=model,
                contents=text.strip()
            )
            
            if not result or not hasattr(result, 'embeddings') or not result.embeddings:
                logger.error(f"Invalid embedding response for text: {text[:50]}...")
                raise InvalidResponseError("Invalid embedding response from Gemini API")
            
            # Extract the embedding vector
            if hasattr(result.embeddings, 'values'):
                embedding_vector = result.embeddings.values
            elif isinstance(result.embeddings, list) and len(result.embeddings) > 0:
                # Handle case where embeddings is a list
                embedding_data = result.embeddings[0]
                if hasattr(embedding_data, 'values'):
                    embedding_vector = embedding_data.values
                else:
                    embedding_vector = embedding_data
            else:
                embedding_vector = result.embeddings
            
            # Convert to numpy array and validate
            embedding_array = np.array(embedding_vector, dtype=np.float32)
            
            if embedding_array.size == 0:
                logger.error(f"Empty embedding vector for text: {text[:50]}...")
                raise InvalidResponseError("Empty embedding vector from Gemini API")
            
            embeddings_list.append(embedding_array)
            
            logger.debug(f"Generated embedding for text (len={len(text)}), embedding dim={len(embedding_array)}")
    
    except ClientError as e:
        error_message = str(e).lower()
        if "rate limit" in error_message or "quota" in error_message:
            logger.error(f"Gemini Embedding API Rate Limit Error: {e}")
            raise GeminiRateLimitError(f"Gemini Embedding API Rate Limit Error: {e}")
        elif "connection" in error_message or "network" in error_message:
            logger.error(f"Gemini Embedding API Connection Error: {e}")
            raise GeminiConnectionError(f"Gemini Embedding API Connection Error: {e}")
        else:
            logger.error(f"Gemini Embedding API Error: {e}")
            raise InvalidResponseError(f"Gemini Embedding API Error: {e}")
    except Exception as e:
        logger.error(f"Gemini Embedding API Call Failed: {e}")
        raise InvalidResponseError(f"Gemini Embedding API Call Failed: {e}")
    
    if not embeddings_list:
        raise InvalidResponseError("No embeddings generated")
    
    # Stack all embeddings into a 2D array
    try:
        embeddings_array = np.vstack(embeddings_list)
        logger.debug(f"Generated {len(embeddings_list)} embeddings with shape {embeddings_array.shape}")
        return embeddings_array
    except Exception as e:
        logger.error(f"Failed to stack embeddings: {e}")
        raise InvalidResponseError(f"Failed to process embeddings: {e}")

# Wrapper function for different Gemini embedding models
async def gemini_embed_experimental(texts: list[str], api_key: str = None) -> np.ndarray:
    """Use the experimental Gemini embedding model."""
    return await gemini_embed(texts, model="gemini-embedding-exp-03-07", api_key=api_key)

async def gemini_embed_latest(texts: list[str], api_key: str = None) -> np.ndarray:
    """Use the latest Gemini embedding model (update model name as needed)."""
    return await gemini_embed(texts, model="gemini-embedding-exp-03-07", api_key=api_key)