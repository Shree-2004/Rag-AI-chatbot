"""
src/llm.py
─────────────────────────────────────────────────────────────────────────────
Responsibility: Return a LangChain chat model based on the provider set in
config.py.  Supports streaming so tokens flow to Streamlit in real time.

Why abstract the LLM?
  pipeline.py and app.py call get_llm() and never import provider-specific
  classes directly.  Switching Gemini → OpenAI is a one-line config change.
─────────────────────────────────────────────────────────────────────────────
"""

from langchain_core.language_models.chat_models import BaseChatModel

import config


def get_llm(streaming: bool = True) -> BaseChatModel:
    """
    Factory: return the configured LLM instance.

    Args:
        streaming: If True, enable token-by-token streaming.
                   Set False for batch/non-interactive use.

    Returns:
        A LangChain BaseChatModel — compatible with LCEL (|) chains and
        .stream() / .invoke() calls.

    Raises:
        ValueError: if LLM_PROVIDER is not recognised.
        EnvironmentError: if the required API key is missing.
    """
    provider = config.LLM_PROVIDER.lower()

    if provider == "gemini":
        return _build_gemini(streaming)
    elif provider == "openai":
        return _build_openai(streaming)
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: '{provider}'. "
            "Choose 'gemini' or 'openai'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Private builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_gemini(streaming: bool) -> BaseChatModel:
    """
    Google Gemini via langchain-google-genai.

    Default model: gemini-1.5-flash (fast, generous free tier).
    Swap to gemini-1.5-pro in config.py for longer context / higher quality.

    Requires: langchain-google-genai, google-generativeai
    """
    if not config.GOOGLE_API_KEY:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. "
            "Add it to your .env file or set it as an environment variable."
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=config.LLM_TEMPERATURE,
        max_output_tokens=config.LLM_MAX_TOKENS,
        streaming=streaming,
        convert_system_message_to_human=True,  # Gemini requires this
    )


def _build_openai(streaming: bool) -> BaseChatModel:
    """
    OpenAI ChatGPT via langchain-openai.

    Default model: gpt-4o-mini (fast, cheap, very capable).
    Swap to gpt-4o in config.py for maximum quality.

    Requires: langchain-openai, openai
    """
    if not config.OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file or set it as an environment variable."
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=config.OPENAI_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        streaming=streaming,
    )