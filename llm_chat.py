"""
LLM Chat Module for TC2-BBS Meshtastic

Provides AI chat functionality with:
- Language detection for token limit optimization
- Conversation history management
- Meshtastic byte-limit validation
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field


class Language(str, Enum):
    """Supported languages for chat."""
    ENGLISH = "english"
    RUSSIAN = "russian"
    GEORGIAN = "georgian"
    UNKNOWN = "unknown"


class LanguageDetection(BaseModel):
    """Structured output for language detection."""
    language: Language = Field(
        description="Detected language of the input text"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score 0.0-1.0"
    )


@dataclass
class LanguageConfig:
    """Configuration for language-specific limits."""
    max_bytes: int
    max_tokens: int
    bytes_per_char: float


# Token limits based on Meshtastic 200-byte constraint
LANGUAGE_LIMITS: dict[str, LanguageConfig] = {
    "english": LanguageConfig(max_bytes=200, max_tokens=55, bytes_per_char=1.0),
    "russian": LanguageConfig(max_bytes=200, max_tokens=35, bytes_per_char=2.0),
    "georgian": LanguageConfig(max_bytes=200, max_tokens=22, bytes_per_char=3.0),
    "unknown": LanguageConfig(max_bytes=200, max_tokens=22, bytes_per_char=3.0),
}


def detect_language(text: str, client: OpenAI, model: str) -> Language:
    """
    Detect the language of input text using LLM.

    Args:
        text: Input text to analyze
        client: OpenAI client instance
        model: Model name for detection

    Returns:
        Detected Language enum value
    """
    try:
        instructor_client = instructor.from_openai(client)

        result = instructor_client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": f"Detect the language of this text: {text}"
            }],
            response_model=LanguageDetection,
        )

        return result.language
    except Exception as e:
        logging.warning(f"Language detection failed: {e}, defaulting to unknown")
        return Language.UNKNOWN


def validate_message_length(text: str, max_bytes: int = 200) -> tuple[str, bool]:
    """
    Ensure message fits within Meshtastic byte limit.

    Args:
        text: Message text to validate
        max_bytes: Maximum allowed bytes (default 200)

    Returns:
        Tuple of (validated_text, was_truncated)
    """
    encoded = text.encode('utf-8')

    if len(encoded) <= max_bytes:
        return text, False

    # Truncate by characters until within byte limit
    truncated = text
    while len(truncated.encode('utf-8')) > max_bytes - 3:  # Reserve 3 bytes for "..."
        truncated = truncated[:-1]

    return truncated.rstrip() + "...", True


class ChatHandler:
    """
    Handles AI chat conversations for Meshtastic BBS.

    Manages conversation context, language detection, and response generation
    with appropriate token limits for each language.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize the chat handler.

        Args:
            api_key: OpenAI API key
            model: Model name for chat completions
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.logger = logging.getLogger("LLM-Chat")

    def generate_response(
        self,
        user_message: str,
        conversation_history: list[dict],
        detected_language: Optional[Language] = None
    ) -> tuple[str, Language]:
        """
        Generate an AI response for the user message.

        Args:
            user_message: The user's input message
            conversation_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
            detected_language: Pre-detected language (if None, will detect)

        Returns:
            Tuple of (response_text, detected_language)
        """
        # Detect language if not provided
        if detected_language is None:
            detected_language = detect_language(user_message, self.client, self.model)

        # Get token limit for detected language
        config = LANGUAGE_LIMITS.get(detected_language.value, LANGUAGE_LIMITS["unknown"])

        self.logger.info(f"Language: {detected_language.value}, max_tokens: {config.max_tokens}")

        # Build system prompt
        system_prompt = f"""You are Groot, a helpful assistant on a Meshtastic mesh network BBS in Tbilisi, Georgia.
CRITICAL CONSTRAINTS:
- Your response MUST be under {config.max_tokens} tokens
- Respond in {detected_language.value}
- Be extremely concise - Twitter/SMS style
- No markdown, no bullet points, just plain text
- User can send R to reset conversation history"""

        # Build messages list
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (limit to last 10 exchanges)
        history_limit = 10
        for msg in conversation_history[-history_limit:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=config.max_tokens,
            )

            raw_content = response.choices[0].message.content

            if raw_content is None:
                self.logger.warning("Response content is None!")
                return "No response from AI.", detected_language

            response_text = raw_content.strip()

            # Validate and truncate if needed
            response_text, was_truncated = validate_message_length(response_text, config.max_bytes)

            if was_truncated:
                self.logger.warning(f"Response truncated to fit {config.max_bytes} bytes")

            return response_text, detected_language

        except Exception as e:
            self.logger.error(f"Chat API error: {e}")
            raise

    def process_message(
        self,
        sender_id: str,
        message: str,
        get_history_func,
        add_message_func,
        clear_history_func
    ) -> str:
        """
        Process an incoming chat message.

        Args:
            sender_id: The sender's node ID
            message: The user's message
            get_history_func: Function to get conversation history from DB
            add_message_func: Function to add message to DB
            clear_history_func: Function to clear conversation history

        Returns:
            Response text to send back
        """
        # Handle empty message
        if not message.strip():
            return "Send message. X=exit, R=reset, H=help"

        try:
            # Get conversation history from database
            history = get_history_func(sender_id)

            # Generate response
            response, language = self.generate_response(message, history)

            # Store user message in history
            add_message_func(sender_id, "user", message, language.value)

            # Store assistant response in history
            add_message_func(sender_id, "assistant", response, language.value)

            return response

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return "Error. Try again."


# Singleton instance (initialized when config is loaded)
_chat_handler: Optional[ChatHandler] = None


def get_chat_handler(api_key: str, model: str = "gpt-4o-mini") -> ChatHandler:
    """
    Get or create the chat handler singleton.

    Args:
        api_key: OpenAI API key
        model: Model name for chat completions

    Returns:
        ChatHandler instance
    """
    global _chat_handler
    if _chat_handler is None:
        _chat_handler = ChatHandler(api_key, model)
    return _chat_handler
