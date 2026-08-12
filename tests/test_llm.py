"""Tests for src/llm.py"""

from unittest.mock import patch, MagicMock

import pytest

import config
from src.llm import get_llm


def test_unknown_provider_raises():
    with patch.object(config, "LLM_PROVIDER", "claude"):
        with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
            get_llm()


def test_openai_missing_key_raises():
    with patch.object(config, "LLM_PROVIDER", "openai"):
        with patch.object(config, "OPENAI_API_KEY", None):
            with pytest.raises(EnvironmentError, match="OPENAI_API_KEY is not set"):
                get_llm()


def test_gemini_missing_key_raises():
    with patch.object(config, "LLM_PROVIDER", "gemini"):
        with patch.object(config, "GOOGLE_API_KEY", None):
            with pytest.raises(EnvironmentError, match="GOOGLE_API_KEY is not set"):
                get_llm()


@patch("src.llm._build_openai")
def test_openai_provider_selected(mock_build):
    mock_build.return_value = MagicMock()
    with patch.object(config, "LLM_PROVIDER", "openai"):
        result = get_llm(streaming=True)
    mock_build.assert_called_once_with(True)
    assert result is mock_build.return_value


@patch("src.llm._build_gemini")
def test_gemini_provider_selected(mock_build):
    mock_build.return_value = MagicMock()
    with patch.object(config, "LLM_PROVIDER", "gemini"):
        result = get_llm(streaming=False)
    mock_build.assert_called_once_with(False)
    assert result is mock_build.return_value
