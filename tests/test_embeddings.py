"""Tests for src/embeddings.py"""

from unittest.mock import patch, MagicMock

import pytest

import config
from src.embeddings import get_embedding_model


def test_unknown_provider_raises():
    with patch.object(config, "EMBEDDING_PROVIDER", "unsupported"):
        with pytest.raises(ValueError, match="Unknown EMBEDDING_PROVIDER"):
            get_embedding_model()


@patch("src.embeddings._get_openai_embeddings")
def test_openai_provider_selected(mock_openai):
    mock_openai.return_value = MagicMock()
    with patch.object(config, "EMBEDDING_PROVIDER", "openai"):
        result = get_embedding_model()
    mock_openai.assert_called_once()
    assert result is mock_openai.return_value


@patch("src.embeddings._get_huggingface_embeddings")
def test_huggingface_provider_selected(mock_hf):
    mock_hf.return_value = MagicMock()
    with patch.object(config, "EMBEDDING_PROVIDER", "huggingface"):
        result = get_embedding_model()
    mock_hf.assert_called_once()
    assert result is mock_hf.return_value


def test_openai_missing_key_raises():
    with patch.object(config, "EMBEDDING_PROVIDER", "openai"):
        with patch.object(config, "OPENAI_API_KEY", None):
            with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
                get_embedding_model()
